## Pretrained checkpoint

A pretrained teacher checkpoint (`v2` backbone, patch size 8, `paris4` run, step 352500;
215.9M-param backbone) is published on HuggingFace:
**[scrollprize/dinovol_v2_ps8_with_paris4_352500](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500)**
— part of the [Representation collection](https://huggingface.co/collections/scrollprize/representation-67e1b44299d5c18f5845874f).

The repo provides a slim teacher-backbone file for inference and the full training
checkpoint for resuming. Load it with
`dinovol_2/eval/embedding_utils.py::load_backbone_from_checkpoint` (the model config
travels inside the weights, so the architecture is rebuilt automatically).

---

an attempt at a faithful implementation of dinov2-style pretraining on 3d volumes. 

- the dinov2_eva is from [dynamic-network-architectures](github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py) , with some minimal changes
- the augmentation library is a loosely modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2)
- normalization is mostly borrowed from [nnunetv2](https://github.com/MIC-DKFZ/nnUNet)
- rope is from the [dinov3 impl](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py), extended to support 3d


this implementation is still incomplete. pretraining works but no finetuning yet written. 

NOTE: a newer `v2` backbone config exists and should generally be preferred for new runs, but the default remains the older config so older checkpoints continue to load without config changes
To select the newer defaults explicitly, set `model.model_type` to `v2` in the config:

```json
{
  "model": {
    "model_type": "v2",
    "embedding_type": "default",
    "global_crops_size": [96, 96, 96],
    "local_crops_size": [48, 48, 48]
  }
}
```

## Gram Anchoring And HR Adaptation

`pretrain.py` now supports the three-stage DINOv3-style workflow:

- base pretraining with DINO + iBOT + KoLeo
- late dense-feature refinement with Gram anchoring
- short mixed-resolution HR adaptation with Gram anchoring kept on

The new config surface is:

- top-level `gram`
  - `enabled`
  - `loss_weight`
  - `teacher_checkpoint`
  - `teacher_refresh_every`
  - `teacher_refresh_start_step`
  - `normalized`
  - `img_level`
  - `remove_neg`
  - `remove_only_teacher_neg`
- dataset keys
  - `gram_teacher_crop_size`
  - `gram_teacher_no_augmentations`
  - `variants`

When `gram.enabled=true`, the trainer builds a frozen Gram teacher backbone, loads it from `gram.teacher_checkpoint` when provided, refreshes it from the live EMA teacher on the configured cadence, and adds an image-level Gram loss on patch features. Gram-teacher crops are paired with each global crop from the exact same sampled 3D region, and default to normalization-only.

Mixed-resolution HR adaptation is enabled by adding `dataset.variants`, where each variant defines its own crop sizes and sampling ratio. The trainer builds one dataloader per variant and samples them with the configured weights. For `embedding_type=deeper`, the dataset automatically derives the needed overscanned `*_view_size` values from the patch halo.

`dinovol_2/example_config.json` remains runnable as a base-pretraining config and also includes a `recipes` object with three complete examples:

- `recipes.base_pretrain`
- `recipes.gram_refinement`
- `recipes.hr_adaptation`

The important stage-specific overrides are:

```json
{
  "gram": {
    "enabled": true,
    "loss_weight": 2.0,
    "teacher_checkpoint": "/path/to/previous/checkpoint.pt",
    "teacher_refresh_every": 10000
  },
  "model": {
    "pretrained_weights": "/path/to/previous/checkpoint.pt",
    "pretrained_backbone_only": false
  }
}
```

For HR adaptation, add weighted crop variants:

```json
{
  "dataset": {
    "variants": [
      {
        "ratio": 0.3,
        "global_crop_size": [128, 128, 128],
        "local_crop_size": [48, 48, 48],
        "gram_teacher_crop_size": [160, 160, 160]
      },
      {
        "ratio": 0.7,
        "global_crop_size": [160, 160, 160],
        "local_crop_size": [80, 80, 80],
        "gram_teacher_crop_size": [192, 192, 192]
      }
    ]
  }
}
```

To sanity-check one batch end to end:

```bash
uv run python -m dinovol_2.verify dinovol_2/example_config.json --no-amp
```

To run the synthetic smoke suite:

```bash
uv run python -m unittest tests.test_pretrain_smoke -v
```

## Patch-4 Multi-Node Pretraining

Patch-4 crops are supported through the mesh strategy intended for H100-class
multi-node runs. The production path uses data parallelism, context parallelism,
tensor parallelism, FSDP2 state sharding, activation checkpointing, distributed
KoLeo, capped/chunked iBOT, mixed RoPE, and `window_global_3d` attention.

Relevant config keys:

- `amp_dtype`: `auto`, `bf16`, `fp16`, or `off`
- `parallelism.strategy`: `ddp`, `tp_ddp`, or `mesh`
- `parallelism.tensor_parallel_size`: tensor-parallel group size
- `parallelism.context_parallel_size`: context-parallel group size
- `parallelism.sharding`: `none`, `fsdp2`, or `zero1`
- `koleo.distributed`: gather CLS tokens across data-parallel ranks before KoLeo
- `ibot.max_masked_patches_per_rank`: absolute cap on selected masked patches
- `ibot.projection_chunk_size` and `ibot.loss_chunk_size`: iBOT memory controls
- `distributed_timeout_seconds`: process-group timeout for slower multi-node starts

The current 4-node production template for patch-4 320³/160³ crops is
`configs/patch4_window320_h100_4node_template.json`. It uses environment
placeholders for outputs, dataset manifests, and W&B run naming, so host names,
credentials, storage URLs, and machine-local paths stay outside the repository.

Synthetic profiler smoke setup:

```bash
RUN_DIR="$PWD/.scratch/dinovol_profile"
mkdir -p "$RUN_DIR"
uv run python scripts/create_synthetic_zarr.py "$RUN_DIR/synthetic.zarr" --shape 224
OUTPUT_DIR="$RUN_DIR/output" SYNTHETIC_ZARR="$RUN_DIR/synthetic.zarr" \
  uv run python -m dinovol_2.profile_pretrain configs/synthetic_patch4_smoke.json --steps 1 --no-resume
```

Multi-node launch template:

```bash
CONFIG=configs/patch4_window320_h100_4node_template.json \
NNODES="$NNODES" NODE_RANK="$NODE_RANK" MASTER_ADDR="$MASTER_ADDR" \
  scripts/launch_multinode_pretrain.sh
```

For c10d rendezvous on private IPs, pass `RDZV_CONF=is_host=true,read_timeout=300`
on node rank 0 and `RDZV_CONF=is_host=false,read_timeout=300` on the other nodes.
This avoids relying on torchrun's hostname/IP host-election heuristic.

For RDMA/RoCE launches, keep `NCCL_IB_DISABLE` unset and pass HCA/GID selection
from the runtime environment. On the tested 4-node H100 allocation, the stable
single rail was:

```bash
NCCL_SOCKET_IFNAME=eth0 \
GLOO_SOCKET_IFNAME=eth0 \
NCCL_IB_HCA=mlx5_2 \
NCCL_IB_GID_INDEX=3 \
NCCL_IB_ADDR_FAMILY=AF_INET \
NCCL_IB_ROCE_VERSION_NUM=2 \
REQUIRE_NCCL_RDMA_ENV=1
```

Run the collective probe before a full training launch on every new allocation:

```bash
CONFIG_REQUIRED=0 MODULE=dinovol_2.collective_probe \
NNODES="$NNODES" NODE_RANK="$NODE_RANK" MASTER_ADDR="$MASTER_ADDR" \
RDZV_CONF="$RDZV_CONF" NCCL_IB_HCA="$NCCL_IB_HCA" NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  scripts/launch_multinode_pretrain.sh --min-bytes 1048576 --max-bytes 67108864 --steps 5
```

For profiling under torchrun, set `MODULE=dinovol_2.profile_pretrain` and pass
profile arguments after the script name. Patch-3 and patch-2 remain research
modes and require separate validation before production training.

## Optional Task Eval During Pretraining

`pretrain.py` can optionally run small downstream segmentation trainings during pretraining.

- set `task_eval_every` to a positive step cadence to enable it
- choose `eval_task` as `both`, `surfaces`, or `ink`
- set `eval_task_train_iters` to control the mini-training length, default `500`
- set `eval_task_decoder_type` to `simple` or `patch_encode_decode`

The task data is downloaded with `python -m dinovol_2.eval.download_data --task both`.

- `both` now means `surfaces` plus `ink`
- `surfaces` is resized 2x before crops are drawn
- `surfaces` and `ink` each use the first 10 sorted samples as the deterministic validation set
- `ink` is not resized before crops are drawn
- train and validation crops are taken from precomputed chunks that contain some foreground and at least 50% background in supervised voxels
- the saved validation image contains one row per validation sample, with image / label / prediction panels
- for `ink`, voxels with `supervision_mask == 0` are ignored and supervised unlabeled voxels are treated as background
- for `ink`, loss/metrics and saved previews use a max projection across Z to match the flat ink trainer

## Napari visualizer

There is a small napari helper for checkpoint inspection at `dinovol_2/eval/napari_visualizer.py`.

Run it with:

```bash
python -m dinovol_2.eval.napari_visualizer
```

Workflow:

- open an OME-Zarr from the widget, click `Load Scales`, choose the desired scale, and click `Open Zarr`
- draw a rectangle in the generated `*_bbox` shapes layer; this 2D YX bbox is applied across the full Z span of the selected scale
- add one or more points in a `Points` layer
- choose a `pretrain.py` checkpoint, image layer, and points layer in the dock widget
- click `Cache Embeddings`
- click `Show Feature PCA` to render a 3-channel PCA view of the cached patch embeddings
- optionally enable `Otsu Foreground Mask` and set `Mask Dilation` before creating the PCA layer
- click `Similarity For Selected Points` or `Similarity For All Points`

The widget rebuilds the teacher backbone from the saved checkpoint config, computes a patch embedding grid only inside the active bbox for the selected OME-Zarr scale, and limits the PCA and cosine-similarity outputs to that same crop. The dock widget opens on the bottom of the napari window.
