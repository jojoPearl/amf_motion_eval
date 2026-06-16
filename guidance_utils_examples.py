"""
Integration example: Wan AMF guidance with modularized components.

Demonstrates:
1. Reference AMF extraction with feature storage
2. Latent optimization with AMF loss
3. K/V injection for zero-shot motion transfer
4. Memory optimization for large-scale generation
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "amf_motion_eval" / "wan_hooks"))
sys.path.insert(0, str(PROJECT_ROOT / "amf_motion_eval"))

# Example 1: Extract reference AMF
def example_extract_amf():
    """Extract reference motion from video."""
    from wan_amf import WanAMFExtractor
    
    # Minimal config object (in real usage, from argparse)
    class Config:
        mode = "extract"
        task = "ti2v-5B"
        ckpt_dir = str(PROJECT_ROOT / "Wan2.2" / "Wan2.2-TI2V-5B")
        video_path = "path/to/reference/video.mp4"
        frame_num = 81
        size = "1280*704"
        guidance_blocks = [15]
        device_id = 0
        output_path = str(PROJECT_ROOT / "outputs" / "wan_amf")
        reference_prompt = ""
        target_prompt = ""
        reference_timestep = "first"
        motion_temp = 2.0
        argmax_motion_flow = True
        head_reduce = "logits"
        sample_solver = "unipc"
        sample_steps = 50
        sample_shift = 5.0
        t5_cpu = False
        offload_t5 = False
        convert_model_dtype = False
    
    config = Config()
    extractor = WanAMFExtractor(config)
    extractor.load_attn_features()
    
    print("✓ Reference AMF extracted and saved via WanFeatureStorage")


# Example 2: K/V snapshot for zero-shot transfer
def example_kv_injection_snapshot():
    """Demonstrate K/V snapshot saving for zero-shot transfer."""
    from guidance_utils import WanFeatureStorage
    
    storage = WanFeatureStorage("outputs/wan_amf")
    
    # In real usage, this would be populated during generation with --enable_kv_injection
    kv_snapshot = {
        1: {
            "key": torch.randn(1, 32, 64, 64),
            "value": torch.randn(1, 32, 64, 64),
            "seq_lens": torch.tensor([32]),
        },
        5: {
            "key": torch.randn(1, 32, 64, 64),
            "value": torch.randn(1, 32, 64, 64),
            "seq_lens": torch.tensor([32]),
        },
        15: {
            "key": torch.randn(1, 32, 64, 64),
            "value": torch.randn(1, 32, 64, 64),
            "seq_lens": torch.tensor([32]),
        },
    }
    
    metadata = {
        "video_path": "reference.mp4",
        "seed": 42,
        "reference_prompt": "a person running",
    }
    
    snapshot_path = storage.save_kv_snapshot(
        kv_dict=kv_snapshot,
        snapshot_name="run_ref_optimal",
        metadata=metadata,
    )
    
    print(f"✓ K/V snapshot saved: {snapshot_path}")
    
    # Later, load and reuse for new prompt
    loaded_snapshot = storage.load_kv_snapshot("run_ref_optimal")
    print(f"✓ K/V snapshot loaded with {len(loaded_snapshot['kv_dict'])} block pairs")


# Example 3: Modularized motion flow computation
def example_motion_flow_computation():
    """Use modularized AMF computation."""
    from guidance_utils import compute_wan_motion_flow, compute_wan_motion_flow_loss
    
    # Mock data (in real usage from actual forward pass)
    batch_size, seq_len, num_heads, head_dim = 1, 441, 32, 64
    grid_size = (9, 7, 7)  # frames, height, width
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cpu")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cpu")
    
    # Compute AMF
    amf = compute_wan_motion_flow(
        q, k,
        grid_size=grid_size,
        temp=2.0,
        argmax=True,
        head_reduce="logits",
    )
    
    print(f"✓ Computed AMF shape: {amf.shape}")  # [9, 9, 7, 7, 2]
    
    # Compute loss (would require gradients in real usage)
    loss = compute_wan_motion_flow_loss(
        q, k,
        ref_amf=amf,
        grid_size=grid_size,
        temp=2.0,
        threshloss=True,
        head_reduce="logits",
    )
    
    print(f"✓ Computed AMF loss: {loss:.6f}")


# Example 4: Feature storage operations
def example_feature_storage():
    """Demonstrate feature storage and retrieval."""
    from guidance_utils import WanFeatureStorage
    
    storage = WanFeatureStorage("outputs/wan_amf_test")
    
    # Save AMF
    amf = torch.randn(9, 9, 7, 7, 2)
    config = {"motion_temp": 2.0, "prompt": "running motion"}
    
    amf_path = storage.save_amf(
        amf=amf,
        block_id=15,
        block_name="block_15_self_attn_processor",
        grid_size=(9, 7, 7),
        config_dict=config,
    )
    
    # Load AMF
    loaded_amf = storage.load_amf(block_id=15)
    assert loaded_amf is not None and loaded_amf.shape == amf.shape
    
    print(f"✓ AMF saved and loaded successfully")
    
    # Get storage info
    info = storage.get_storage_info()
    print(f"✓ Storage info: {info}")


if __name__ == "__main__":
    print("=" * 60)
    print("Wan AMF Motion Transfer - Modularized Components Demo")
    print("=" * 60)
    
    print("\n[1] Motion flow computation")
    example_motion_flow_computation()
    
    print("\n[2] Feature storage operations")
    example_feature_storage()
    
    print("\n[3] K/V snapshot management")
    example_kv_injection_snapshot()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
