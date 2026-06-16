"""
AMF (Attention Motion Flow) computation utilities for Wan models.

Implements motion flow extraction from attention patterns and loss computation
for motion-guided video generation.
"""

import math
import torch
import torch.nn.functional as F


def compute_displacement(attn, height, width, argmax=False):
    """Compute pixel displacement from attention map.
    
    Args:
        attn: Attention weights [HW, HW]
        height: Spatial height
        width: Spatial width
        argmax: If True, use hard attention (argmax); else soft (expected value)
        
    Returns:
        Displacement flow [HW, 2] with (dx, dy) for each pixel
    """
    device = attn.device

    if argmax:
        # Hard attention: argmax of attention weights
        matches = attn.argmax(dim=-1)
        source = torch.arange(attn.shape[0], device=device)

        x1 = source % width
        y1 = source // width
        x2 = matches % width
        y2 = matches // width

        return torch.stack((x2 - x1, y2 - y1), dim=-1).float()

    # Soft attention: expected displacement
    dtype = attn.dtype
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )

    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    exp_x = attn @ x_flat
    exp_y = attn @ y_flat

    dx = exp_x - x_flat
    dy = exp_y - y_flat

    return torch.stack((dx, dy), dim=-1)


def _select_wan_video_tokens(q, k, grid_size):
    """Extract video tokens from Q/K.
    
    Wan self-attention saves Q/K as [B, S, heads, D].
    This reshapes to [heads, frames, hw, head_dim] for frame-pair analysis.
    
    Args:
        q: Query [B, S, heads, D]
        k: Key [B, S, heads, D]
        grid_size: (frames, height, width)
        
    Returns:
        Tuple of (q, k, frames, height, width, hw) reshaped appropriately
    """
    frames, height, width = [int(v) for v in grid_size]
    hw = height * width
    seq_len = frames * hw

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"Expected Q/K shape [B,S,H,D], got {q.shape=} {k.shape=}")

    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("Only batch_size=1 supported")

    if q.shape[1] < seq_len or k.shape[1] < seq_len:
        raise ValueError(
            f"Q/K shorter than grid_size: need {seq_len}, "
            f"got q={q.shape[1]}, k={k.shape[1]}"
        )

    q = q[0, :seq_len].permute(1, 0, 2).reshape(
        q.shape[2],
        frames,
        hw,
        q.shape[-1],
    )
    k = k[0, :seq_len].permute(1, 0, 2).reshape(
        k.shape[2],
        frames,
        hw,
        k.shape[-1],
    )

    return q.float(), k.float(), frames, height, width, hw


def compute_pair_attention_from_heads(q_i, k_j, temp=2.0, head_reduce="logits"):
    """Compute attention for one frame pair from per-head Q/K.
    
    Two modes of head reduction:
    - 'logits': average logits across heads, then softmax
    - 'attn': softmax per-head, then average
    
    Args:
        q_i: Query for frame i [num_heads, hw, head_dim]
        k_j: Key for frame j [num_heads, hw, head_dim]
        temp: Temperature for softmax
        head_reduce: Head aggregation mode
        
    Returns:
        Attention map [hw, hw]
    """
    scale = math.sqrt(q_i.shape[-1])
    logits = torch.matmul(q_i, k_j.transpose(-1, -2)) / scale

    if head_reduce == "logits":
        return F.softmax(logits.mean(dim=0) * temp, dim=-1)

    if head_reduce == "attn":
        return F.softmax(logits * temp, dim=-1).mean(dim=0)

    raise ValueError(f"Unsupported head_reduce={head_reduce!r}")


def compute_wan_motion_flow(
    q,
    k,
    grid_size,
    temp=2.0,
    argmax=True,
    head_reduce="logits",
):
    """Compute AMF from saved self-attention Q/K.
    
    Extracts motion flow from frame-to-frame attention patterns.
    
    Args:
        q: Query [B, S, heads, D]
        k: Key [B, S, heads, D]
        grid_size: (frames, height, width)
        temp: Temperature for attention softmax
        argmax: Hard vs soft displacement
        head_reduce: Head aggregation strategy
        
    Returns:
        AMF tensor [F, F, H, W, 2] (frame_i, frame_j, height, width, (dx, dy))
    """
    q, k, frames, height, width, hw = _select_wan_video_tokens(q, k, grid_size)

    flows = []

    for frame_i in range(frames):
        row = []
        q_i = q[:, frame_i]

        for frame_j in range(frames):
            k_j = k[:, frame_j]
            attn = compute_pair_attention_from_heads(
                q_i,
                k_j,
                temp=temp,
                head_reduce=head_reduce,
            )

            flow = compute_displacement(
                attn,
                height,
                width,
                argmax=argmax,
            )

            row.append(flow.reshape(height, width, 2))

        flows.append(torch.stack(row, dim=0))

    return torch.stack(flows, dim=0)


def compute_wan_motion_flow_loss(
    q,
    k,
    ref_amf,
    grid_size,
    temp=2.0,
    threshloss=True,
    head_reduce="logits",
):
    """Compute differentiable AMF loss for latent optimization.
    
    Generated AMF is differentiable (soft attention).
    Reference AMF is detached fixed target.
    
    Args:
        q: Current query [B, S, heads, D]
        k: Current key [B, S, heads, D]
        ref_amf: Reference AMF [F, F, H, W, 2]
        grid_size: (frames, height, width)
        temp: Temperature for softmax
        threshloss: If True, only compute loss on non-zero reference motion
        head_reduce: Head aggregation strategy
        
    Returns:
        Scalar loss
    """
    q, k, frames, height, width, hw = _select_wan_video_tokens(q, k, grid_size)

    total_loss = None
    num_losses = 0

    for frame_i in range(frames):
        q_i = q[:, frame_i]

        for frame_j in range(frames):
            k_j = k[:, frame_j]

            attn = compute_pair_attention_from_heads(
                q_i,
                k_j,
                temp=temp,
                head_reduce=head_reduce,
            )

            flow = compute_displacement(
                attn,
                height,
                width,
                argmax=False,  # Use soft displacement for differentiable loss
            ).reshape(height, width, 2)

            ref_flow = ref_amf[frame_i, frame_j].to(
                device=flow.device,
                dtype=flow.dtype,
            ).detach()

            if threshloss:
                # Only compute loss where reference has motion
                mask = torch.norm(ref_flow, dim=-1) > 0
                if not mask.any():
                    continue
                loss = F.mse_loss(flow[mask], ref_flow[mask])
            else:
                # Full frame MSE loss
                loss = F.mse_loss(flow, ref_flow)

            total_loss = loss if total_loss is None else total_loss + loss
            num_losses += 1

    if total_loss is None:
        return q.new_tensor(0.0)

    return total_loss / max(num_losses, 1)
