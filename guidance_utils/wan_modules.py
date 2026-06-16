"""
Feature storage and snapshot management for Wan motion guidance.

Provides utilities to save, load, and manage AMF snapshots for zero-shot
motion injection across multiple generations.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


class WanFeatureStorage:
    """Manages storage and retrieval of motion features (AMF, K/V snapshots).
    
    Handles:
    - Saving reference AMF to disk
    - Loading cached AMF for multiple generations
    - Managing K/V injection snapshots
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.amf_dir = self.storage_path / "amf_reference"
        self.kv_snapshot_dir = self.storage_path / "kv_snapshots"
        self.metadata_path = self.storage_path / "metadata.json"

        self.amf_dir.mkdir(parents=True, exist_ok=True)
        self.kv_snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or initialize metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_amf(
        self,
        amf: torch.Tensor,
        block_id: int,
        block_name: str,
        grid_size: Tuple[int, int, int],
        config_dict: Dict,
    ) -> str:
        """Save reference AMF tensor and metadata.
        
        Args:
            amf: AMF tensor [F, F, H, W, 2]
            block_id: Block identifier
            block_name: Human-readable block name
            grid_size: (frames, height, width) in latent space
            config_dict: Additional configuration to save
            
        Returns:
            Path to saved AMF file
        """
        safe_block_name = block_name.replace(".", "_").replace("/", "_")
        amf_path = self.amf_dir / f"{safe_block_name}.pt"

        save_dict = {
            "amf": amf.detach().cpu(),
            "block_id": block_id,
            "block_name": block_name,
            "shape": tuple(amf.shape),
            "grid_size": grid_size,
            "patches_height": grid_size[1],
            "patches_width": grid_size[2],
            "latent_num_frames": grid_size[0],
        }
        save_dict.update(config_dict)

        torch.save(save_dict, amf_path)

        # Update metadata
        self.metadata[f"block_{block_id}"] = {
            "block_name": block_name,
            "amf_path": str(amf_path.relative_to(self.storage_path)),
            "shape": tuple(amf.shape),
            "grid_size": grid_size,
        }
        self._save_metadata()

        return str(amf_path)

    def load_amf(self, block_id: int) -> Optional[torch.Tensor]:
        """Load cached AMF tensor for a block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            AMF tensor or None if not found
        """
        key = f"block_{block_id}"
        if key not in self.metadata:
            return None

        amf_path = self.storage_path / self.metadata[key]["amf_path"]
        if not amf_path.exists():
            return None

        checkpoint = torch.load(amf_path, map_location="cpu")
        return checkpoint.get("amf")

    def load_all_amf(self) -> Dict[int, torch.Tensor]:
        """Load all cached AMF tensors.
        
        Returns:
            Dict mapping block_id -> AMF tensor
        """
        amf_dict = {}
        for key, meta in self.metadata.items():
            if key.startswith("block_"):
                block_id = int(key.split("_")[1])
                amf = self.load_amf(block_id)
                if amf is not None:
                    amf_dict[block_id] = amf
        return amf_dict

    def save_kv_snapshot(
        self,
        kv_dict: Dict[int, Dict[str, torch.Tensor]],
        snapshot_name: str,
        metadata: Dict,
    ) -> str:
        """Save K/V injection snapshot for zero-shot transfer.
        
        Args:
            kv_dict: Dict mapping block_id -> key/value/seq_lens tensors
            snapshot_name: Unique identifier for this snapshot
            metadata: Associated metadata (prompt, seed, etc.)
            
        Returns:
            Path to saved snapshot
        """
        snapshot_path = self.kv_snapshot_dir / f"{snapshot_name}.pt"

        save_dict = {"kv_dict": kv_dict, "metadata": metadata}
        torch.save(save_dict, snapshot_path)

        return str(snapshot_path)

    def load_kv_snapshot(self, snapshot_name: str) -> Optional[Dict]:
        """Load K/V injection snapshot.
        
        Args:
            snapshot_name: Snapshot identifier
            
        Returns:
            Dict with 'kv_dict' and 'metadata', or None if not found
        """
        snapshot_path = self.kv_snapshot_dir / f"{snapshot_name}.pt"
        if not snapshot_path.exists():
            return None

        return torch.load(snapshot_path, map_location="cpu")

    def list_kv_snapshots(self) -> list:
        """List all available K/V snapshots."""
        snapshots = []
        for file in self.kv_snapshot_dir.glob("*.pt"):
            snapshots.append(file.stem)
        return sorted(snapshots)

    def get_storage_info(self) -> Dict:
        """Get information about stored features."""
        amf_files = list(self.amf_dir.glob("*.pt"))
        kv_snapshots = self.list_kv_snapshots()

        return {
            "storage_path": str(self.storage_path),
            "amf_blocks": len(amf_files),
            "kv_snapshots": len(kv_snapshots),
            "metadata_entries": len(self.metadata),
        }
