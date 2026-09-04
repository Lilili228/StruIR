# StruIR

Official PyTorch implementation of **StruIR: Structural Prior Transfer with an Intensity-Statistics-Conditioned Mamba Discriminator for Visible-to-Infrared Image Translation**.

StruIR is a two-stage visible-to-infrared (VI-to-IR) image translation framework designed to preserve scene structure while improving the agreement between generated and real infrared images.

In Stage 1, masked reconstruction and cross-modal contrastive learning are jointly employed to learn structural priors shared by paired visible and infrared images. In Stage 2, the pretrained visible encoder initializes the generator, while Cross-Layer Skip Connections (CL-Skip) equipped with Cross-level Selective Attention Fusion (CSAF) transfer and refine multiscale structural features.

The generator is supervised by two complementary discriminators: a U-Net-based CNN discriminator for local edges, textures, and intensity transitions, and an Intensity-Statistics-Conditioned Mamba Discriminator (ISC-MD) that uses image-level infrared intensity statistics to condition long-range feature modeling.

## Highlights

- A two-stage framework for visible-to-infrared image translation.
- Cross-modal structural-prior learning through masked reconstruction and contrastive representation alignment.
- Transfer of the pretrained visible encoder to provide structure-aware initialization for infrared image generation.
- Cross-Layer Skip Connections for transferring multiscale structural information between the encoder and decoder.
- Cross-level Selective Attention Fusion for adaptively integrating adjacent shallow and deep features.
- A dual-discriminator design combining local convolutional supervision with image-wide long-range adversarial assessment.
- An Intensity-Statistics-Conditioned Mamba Discriminator that encodes infrared mean, standard deviation, and a soft histogram into a conditioning token.
- Evaluation support for pixel-level, structural, perceptual, and distribution-based image-quality metrics.

## Framework

StruIR consists of two training stages.

### Stage 1: Cross-Modal Structural-Prior Learning

Two modality-specific encoder-decoder branches process paired visible and infrared images. The same spatial mask is applied to both modalities, forcing the encoders to recover contours, region boundaries, object layouts, and other structural information from incomplete observations.

Masked reconstruction is optimized together with an NT-Xent cross-modal contrastive objective at the encoder bottleneck. The reconstruction decoders mainly use intermediate and deep features, reducing the risk of solving the pretraining task through shallow low-level shortcuts.

After pretraining, only the visible-image encoder is transferred to Stage 2. The reconstruction decoders and projection heads are discarded.

### Stage 2: Structural-Prior-Guided Generation

The pretrained visible encoder initializes the encoder of the VI-to-IR generator. The transferred encoder remains trainable and is jointly optimized with the decoder.

CL-Skip modules are used in the intermediate and shallow skip pathways. Each CL-Skip employs CSAF to align and selectively fuse the current-scale feature with adjacent shallow and deep features. This allows deeper semantic information to guide the recovery of boundaries and spatial details while reducing the direct transmission of visible-modality-specific textures. The deepest skip pathways retain direct same-scale connections.

The generated infrared image is evaluated by two complementary discriminators:

- A conditional U-Net-based CNN discriminator that focuses on local textures, edges, intensity transitions, and correspondence with the visible input.
- ISC-MD, which incorporates image-level infrared intensity statistics into Mamba-based long-range feature modeling for complementary image-wide adversarial supervision.

Only the generator is required during inference.

## Core Components

### Cross-level Selective Attention Fusion

CSAF aligns adjacent shallow and deep features to the current scale and learns normalized selection weights for adaptive fusion. A channel-spatial attention branch further refines the fused representation. Residual scaling parameters are initialized to zero so that training begins from the original current-scale feature and gradually introduces cross-level information.

### Intensity-Statistics-Conditioned Mamba Discriminator

ISC-MD receives the concatenated visible-infrared pair. For the infrared image, it extracts:

- Global mean intensity
- Sample standard deviation
- A 64-bin differentiable soft histogram over `[-1, 1]`

The resulting 66-dimensional statistic vector is projected to a 128-dimensional conditioning token. Image patches are embedded into the same feature dimension and processed by four gated SS2D blocks. The statistics token modulates long-range spatial modeling in every block before the final discriminator prediction is produced.

### Training Objectives

Stage 1 uses:

- Visible-image masked reconstruction loss
- Infrared-image masked reconstruction loss
- Cross-modal NT-Xent contrastive loss

Stage 2 uses:

- Local U-Net adversarial loss
- ISC-MD adversarial loss
- L1 reconstruction loss
- SSIM loss

The Stage-2 generator objective is:

```text
L_G = L_adv + 10 × L1 + 10 × (1 - SSIM)
```

The local and ISC-MD adversarial terms are weighted by `0.8` and `0.2`, respectively.

## Repository Structure

```text
StruIR/
|-- Stage-1/
|   |-- train_vir2ir.py              # Stage-1 training entry point
|   `-- masking_vir2ir.py            # Dual-modal masked reconstruction model
|-- data/
|   |-- custom_dataset_data_loader.py
|   `-- thermal_dataset.py           # Paired visible-infrared dataset loader
|-- models/
|   |-- struir_model.py              # Main StruIR training model and objectives
|   |-- networks.py                  # Generator and shared network components
|   |-- CSAF.py                      # CSAF implementation
|   |-- base_model.py                # Checkpoint and learning-rate utilities
|   `-- unetgan/
|       |-- ISCMD.py                 # ISC-MD and SS2D implementation
|       `-- unet_discriminator.py    # Local U-Net discriminator
|-- options/
|   |-- base_options.py
|   |-- train_options.py
|   `-- test_options.py
|-- lpips/                           # LPIPS implementation and AlexNet weights
|-- util/                            # Visualization and image utilities
|-- train.py                         # Stage-2 training entry point
|-- test.py                          # Image generation and qualitative testing
|-- evaluate.py                      # Quantitative evaluation
|-- eval.py                          # Evaluation helper used by the training code
|-- ssim.py                          # SSIM implementation
`-- requirements.txt                # Python dependencies
```

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
conda create -n struir python=3.10 -y
conda activate struir
```

Install a PyTorch build compatible with your CUDA version, followed by the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

The main dependencies include PyTorch, TorchVision, Mamba-SSM, timm, NumPy, Pillow, SciPy, OpenCV, Visdom, TensorFlow, and pytorch-fid. LPIPS code and AlexNet v0.1 weights are included in this repository.

## Dataset Preparation

The paired-image loader supports the following dataset modes:

| Dataset | Command-line value |
|---|---|
| FMB | `--dataset_mode FMB` |
| VEDAI | `--dataset_mode VEDAI` |
| AVIID-1 | `--dataset_mode AVIID_1` |
| DRTV30K | `--dataset_mode DRTV30K` |
| M3FD | `--dataset_mode M3FD` |

Organize each dataset as follows:

```text
datasets/DATASET_NAME/
|-- train/
|   |-- 000001_vis.png
|   |-- 000001_ir.png
|   |-- 000002_vis.png
|   `-- 000002_ir.png
`-- test/
    |-- 000101_vis.png
    |-- 000101_ir.png
    |-- 000102_vis.png
    `-- 000102_ir.png
```

Each visible-infrared pair must share the same sample identifier:

```text
<sample_id>_vis.png
<sample_id>_ir.png
```

Visible images are loaded as three-channel RGB images, and infrared images are converted to single-channel grayscale. Images are resized to `512 × 512` by default and normalized to `[-1, 1]`.

DRTV30K currently uses the same pairing logic as FMB. Boundary cropping is not performed automatically. If the original `820 × 580` images must be cropped by 50 pixels on each border, complete that preprocessing step before arranging the files in the structure above.

## Stage-1 Pretraining

Run Stage-1 pretraining separately for every target dataset. The following command uses FMB as an example:

```bash
python Stage-1/train_vir2ir.py \
  --data-root ./datasets/FMB \
  --split train \
  --save-dir ./checkpoints/FMB_stage1 \
  --load-size 512 \
  --batch-size 16 \
  --num-epochs 100 \
  --learning-rate 1e-4 \
  --mask-ratio 0.5 \
  --patch-size 16 \
  --save-frequency 5 \
  --gpu-id 0 \
  --seed 42 \
  --amp
```

The `--amp` flag enables CUDA automatic mixed precision and may be omitted when it is not required. Use `--gpu-id -1` only for a small CPU smoke test.

Stage-1 checkpoints are written to:

```text
checkpoints/FMB_stage1/
|-- encoders_epoch_5.pth
|-- encoders_epoch_10.pth
|-- ...
|-- encoders_best.pth
|-- encoders_final.pth
`-- samples/
```

Use `encoders_best.pth` to initialize Stage 2.

## Stage-2 Training

Train StruIR using the pretrained visible encoder:

```bash
python train.py \
  --dataroot ./datasets/FMB \
  --dataset_mode FMB \
  --model struir \
  --name struir_FMB \
  --checkpoints_dir ./checkpoints \
  --which_model_netG unet_512 \
  --which_model_netD unetdiscriminator \
  --batchSize 4 \
  --loadSize 512 \
  --fineSize 512 \
  --lr 0.0002 \
  --niter 100 \
  --niter_decay 100 \
  --lambda_A 10 \
  --save_epoch_freq 5 \
  --gpu_ids 0 \
  --display_id 0
```

If `--pretrained_encoder_path` is omitted, the program automatically looks for:

```text
<checkpoints_dir>/<dataset_mode>_stage1/encoders_best.pth
```

The command above therefore loads:

```text
./checkpoints/FMB_stage1/encoders_best.pth
```

Use an explicit path when the checkpoint is stored elsewhere:

```bash
--pretrained_encoder_path /path/to/encoders_best.pth
```

## Checkpoint Selection

Periodic Stage-2 checkpoints are saved every five epochs. The mean total generator loss is also computed after every epoch, and the best checkpoint is updated whenever this loss decreases.

```text
checkpoints/struir_FMB/
|-- best_net_G.pth
|-- best_net_D.pth
|-- best_net_D_ISCMD.pth
|-- best_checkpoint.pth
|-- latest_net_G.pth
|-- latest_net_D.pth
|-- latest_net_D_ISCMD.pth
|-- 5_net_G.pth
|-- 5_net_D.pth
|-- 5_net_D_ISCMD.pth
`-- ...
```

The `best_checkpoint.pth` file records the best epoch and its mean generator loss.

## Resume Training

The following example assumes that epoch 100 is the latest completed epoch and resumes from epoch 101:

```bash
python train.py \
  --dataroot ./datasets/FMB \
  --dataset_mode FMB \
  --model struir \
  --name struir_FMB \
  --checkpoints_dir ./checkpoints \
  --continue_train \
  --which_epoch latest \
  --epoch_count 101 \
  --batchSize 4 \
  --gpu_ids 0 \
  --display_id 0
```

Set `--epoch_count` to the number of completed epochs plus one. The current resume path restores network weights and the previous best-loss record, but it does not restore optimizer states.

## Inference

Inference loads the best generator checkpoint by default and processes the complete test set when `--how_many` is omitted:

```bash
python test.py \
  --dataroot ./datasets/FMB \
  --dataset_mode FMB \
  --model struir \
  --name struir_FMB \
  --checkpoints_dir ./checkpoints \
  --which_epoch best \
  --results_dir ./results \
  --save_img_dir ./results/struir_FMB/generated \
  --gpu_ids 0
```

The generator is placed in evaluation mode during inference. If `--save_img_dir` is omitted, generated infrared images are saved to:

```text
results/<experiment_name>/test_best/generated/
```

To process only the first N images, add:

```bash
--how_many N
```

## Evaluation

Run quantitative evaluation on the complete test set with:

```bash
python evaluate.py \
  --dataroot ./datasets/FMB \
  --dataset_mode FMB \
  --model struir \
  --name struir_FMB \
  --checkpoints_dir ./checkpoints \
  --which_epoch best \
  --results_dir ./results \
  --gpu_ids 0
```

## Training on Another Dataset

Keep the dataset root, dataset mode, experiment name, and Stage-1 checkpoint directory consistent. For example, train Stage 2 on DRTV30K with:

```bash
python train.py \
  --dataroot ./datasets/DRTV30K \
  --dataset_mode DRTV30K \
  --model struir \
  --name struir_DRTV30K \
  --checkpoints_dir ./checkpoints \
  --batchSize 4 \
  --lr 0.0002 \
  --niter 100 \
  --niter_decay 100 \
  --gpu_ids 0 \
  --display_id 0
```

This command automatically loads:

```text
./checkpoints/DRTV30K_stage1/encoders_best.pth
```
