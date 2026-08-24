# AMF Motion Evaluation

Wan2.2 AMF extraction, motion-guided generation, and attention validation experiments.

## Directory Layout

- `src/amf_motion_eval/shared/`
  Shared AMF primitives: Wan attention hooks, motion-flow computation, and feature snapshot storage.
- `src/amf_motion_eval/extract_motion/`
  Step 1: entrypoints for extracting motion flow from Wan attention.
- `src/amf_motion_eval/generate_transfer/`
  Step 2: AMF-guided video generation and zero-shot motion transfer.
- `src/amf_motion_eval/evaluate_attention/`
  Step 3: annotation, phantom-limb evaluation, support maps, and temporal-difference heatmaps.
- `scripts/`
  Cluster and local workflow entrypoints.
- `configs/`
  Experiment configs and annotated case definitions.
- `docs/`
  Implementation notes and architecture writeups.
- `outputs/`
  Repo-local experiment outputs, ignored by git.

## Main Entry Points

- Extract motion flow:
  `python -m amf_motion_eval.extract_motion.extract_reference_amf`
- Generate or transfer motion:
  `python -m amf_motion_eval.generate_transfer.wan_amf --mode generate`
- Evaluate and visualize attention:
  `python -m amf_motion_eval.evaluate_attention.phantom_limb_eval`
  `python -m amf_motion_eval.evaluate_attention.amf_support_map_viz`
  `python -m amf_motion_eval.evaluate_attention.amf_temporal_diff_viz`

## Phantom Limb Temporal Diff

`amf_temporal_diff_viz` now compares consecutive AMF flows/support maps:

1. load adjacent AMF maps `F_{t->t+1}` and `F_{t+1->t+2}`
2. compute `D_t = abs(F_{t+1->t+2} - F_{t->t+1})`
3. upsample `D_t` to the original frame size
4. overlay `D_t` on the target frame for latent `t+1`
5. find connected high-response regions
6. test whether each high-response region overlaps the person bbox
7. write suspected phantom-limb regions to `metadata/temporal_diff_metadata.json`

Useful outputs live under `outputs/amf_map_viz/<case_id>/temporal_diff/`:

- `diff_sum_upsampled_l_XX.npy` / `diff_max_upsampled_l_XX.npy`
- `diff_sum_detection_l_XX.png` / `diff_max_detection_l_XX.png`
- `metadata/temporal_diff_metadata.json`

Detection thresholds can be tuned with:

```bash
python -m amf_motion_eval.evaluate_attention.amf_temporal_diff_viz \
  --threshold_percentile 90 \
  --min_area_fraction 0.002 \
  --min_person_overlap 0.20 \
  --max_normal_overlap_for_suspect 0.35
```

For one input video, use the shell workflow:

```bash
VIDEO_PATH=/path/to/video.mp4 \
FRAME_NUM=57 \
PERSON_BBOX=350,120,900,650 \
bash amf_motion_eval/scripts/visualize_attention_maps.sh both
```

`PERSON_BBOX` is optional. If it is omitted, the detector uses the full frame as
the person region and reports high-response regions as candidates.
