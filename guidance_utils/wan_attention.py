"""
Wan attention processor for AMF capture and K/V injection.

Provides the monkey-patching infrastructure to intercept attention computations
and capture Q/K for AMF extraction or inject reference K/V during generation.
"""

import types
from enum import Enum
import torch


class ProcessorMode(Enum):
    """Processor state management using mode-based state machine."""
    IDLE = "idle"                          # No capture, no injection
    CAPTURE = "capture"                    # Recording Q/K for reference AMF
    CAPTURE_FOR_GRAD = "capture_for_grad"  # Recording Q/K with gradient enabled
    INJECT = "inject"                      # Using captured K/V for guidance


class WanAMFProcessor:
    """Processor-like state for AMF capture and optional K/V injection.
    
    Manages lifecycle of attention features:
    - CAPTURE: Save Q/K/V from reference video (detached)
    - CAPTURE_FOR_GRAD: Save Q/K/V with gradients for loss computation
    - INJECT: Use saved K/V to guide generation
    - IDLE: No operation
    """

    def __init__(self, block_name):
        self.block_name = block_name
        self.mode = ProcessorMode.IDLE

        # Captured features (reference or current generation)
        self.query = None
        self.key = None
        self.value = None

        # Injection buffers (cached reference K/V)
        self.inject_key = None
        self.inject_value = None
        self.inject_seq_lens = None

        # Metadata
        self.grid_size = None
        self.seq_lens = None

    def set_mode(self, mode):
        """Transition to a new processor mode, clearing incompatible state."""
        if mode == ProcessorMode.IDLE:
            self.clear()
        elif mode == ProcessorMode.CAPTURE:
            self.clear_capture()
            self.clear_injection()
        elif mode == ProcessorMode.CAPTURE_FOR_GRAD:
            self.clear_capture()
            self.clear_injection()
        elif mode == ProcessorMode.INJECT:
            # Keep injected K/V, clear capture state
            self.clear_capture()

        self.mode = mode

    def clear_capture(self):
        """Clear captured Q/K/V from current forward pass."""
        self.query = None
        self.key = None
        self.value = None
        self.grid_size = None
        self.seq_lens = None

    def clear_injection(self):
        """Clear cached injection buffers."""
        self.inject_key = None
        self.inject_value = None
        self.inject_seq_lens = None

    def clear(self):
        """Clear all state."""
        self.clear_capture()
        self.clear_injection()

    def save(self, q, k, v, seq_lens, grid_sizes):
        """Save Q/K/V only if in capture mode.
        
        Args:
            q: Query tensor [B, S, heads, D]
            k: Key tensor [B, S, heads, D]
            v: Value tensor [B, S, heads, D]
            seq_lens: Sequence lengths
            grid_sizes: Grid dimensions
        """
        if self.mode not in (ProcessorMode.CAPTURE, ProcessorMode.CAPTURE_FOR_GRAD):
            return

        if self.mode == ProcessorMode.CAPTURE_FOR_GRAD:
            # Keep gradients for loss computation
            self.query = q
            self.key = k
            self.value = v
        else:
            # Detach for reference storage
            self.query = q.detach()
            self.key = k.detach()
            self.value = v.detach()

        self.seq_lens = seq_lens.detach().cpu() if seq_lens is not None else None
        self.grid_size = grid_sizes[0].detach().cpu() if grid_sizes is not None else None

    def promote_capture_to_injection(self):
        """Move captured K/V to injection buffers for K/V injection mode.
        
        Raises:
            RuntimeError: If no K/V has been captured.
        """
        if self.key is None or self.value is None:
            raise RuntimeError(f"No captured K/V to inject for {self.block_name}")

        self.inject_key = self.key.detach()
        self.inject_value = self.value.detach()
        self.inject_seq_lens = self.seq_lens.clone() if self.seq_lens is not None else None

        self.clear_capture()
        self.set_mode(ProcessorMode.INJECT)

    def set_injection(self, key, value, seq_lens=None):
        """Install cached K/V buffers and enter injection mode."""
        self.inject_key = key.detach()
        self.inject_value = value.detach()
        self.inject_seq_lens = seq_lens.detach() if seq_lens is not None else None
        self.set_mode(ProcessorMode.INJECT)


def set_wan_amf_processor(attn, processor):
    """Patch one WanSelfAttention module with AMF processor.
    
    Replaces the forward method to intercept Q/K/V and perform capture/injection.
    
    Args:
        attn: WanSelfAttention module to patch
        processor: WanAMFProcessor instance
    """
    from wan.modules.attention import flash_attention
    from wan.modules.model import rope_apply

    attn.amf_processor = processor

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """Modified forward with capture/injection."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Project and reshape
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply RoPE
        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        # Save to processor if in capture mode
        proc = self.amf_processor
        proc.save(q_rope, k_rope, v, seq_lens, grid_sizes)

        attn_k = k_rope
        attn_v = v
        attn_lens = seq_lens

        # Inject reference K/V if in inject mode
        if proc.mode == ProcessorMode.INJECT:
            if proc.inject_key is None or proc.inject_value is None:
                raise RuntimeError(
                    f"K/V injection enabled but no cached K/V for {proc.block_name}"
                )

            inj_k = proc.inject_key.to(device=k_rope.device, dtype=k_rope.dtype)
            inj_v = proc.inject_value.to(device=v.device, dtype=v.dtype)

            if inj_k.shape[0] != b or inj_v.shape[0] != b:
                raise ValueError(
                    f"Batch mismatch in {proc.block_name}: "
                    f"inject_batch={inj_k.shape[0]}, current_batch={b}"
                )

            if inj_k.shape[2:] != k_rope.shape[2:] or inj_v.shape[2:] != v.shape[2:]:
                raise ValueError(
                    f"Head/dim mismatch in {proc.block_name}: "
                    f"inject_k={inj_k.shape}, current_k={k_rope.shape}, "
                    f"inject_v={inj_v.shape}, current_v={v.shape}"
                )

            # Concatenate injected and current K/V
            attn_k = torch.cat([inj_k, k_rope], dim=1)
            attn_v = torch.cat([inj_v, v], dim=1)

            if seq_lens is not None:
                if proc.inject_seq_lens is not None:
                    inj_lens = proc.inject_seq_lens.to(
                        device=seq_lens.device,
                        dtype=seq_lens.dtype,
                    )
                else:
                    inj_lens = torch.full_like(seq_lens, inj_k.shape[1])
                attn_lens = seq_lens + inj_lens

        # Compute attention
        x = flash_attention(
            q=q_rope,
            k=attn_k,
            v=attn_v,
            k_lens=attn_lens,
            window_size=self.window_size,
        )

        x = x.flatten(2)
        x = self.o(x)

        return x

    attn.forward = types.MethodType(forward, attn)


def register_wan_amf_processors(model_blocks, guidance_blocks):
    """Register AMF processors on specified attention blocks.
    
    Args:
        model_blocks: List of all transformer blocks
        guidance_blocks: List of block indices to monitor
        
    Returns:
        dict: Mapping block_id -> WanAMFProcessor
    """
    attn_processors = {}

    for block_id in guidance_blocks:
        block_name = f"block_{block_id}_self_attn_processor"
        processor = WanAMFProcessor(block_name=block_name)

        set_wan_amf_processor(
            model_blocks[block_id].self_attn,
            processor,
        )

        attn_processors[block_id] = processor

    return attn_processors
