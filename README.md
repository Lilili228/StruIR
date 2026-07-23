# StruIR

Official PyTorch implementation of **StruIR: Structural Prior Transfer with an Intensity-Statistics-Conditioned Mamba Discriminator for Visible-to-Infrared Image Translation**.

StruIR is a two-stage visible-to-infrared (VI-to-IR) image translation framework designed to preserve scene structure while improving the agreement between generated and real infrared images.

In Stage 1, masked reconstruction and cross-modal contrastive learning are jointly employed to learn structural priors shared by paired visible and infrared images. In Stage 2, the pretrained visible encoder initializes the generator, while Cross-Layer Skip Connections (CL-Skip) equipped with Cross-level Selective Attention Fusion (CSAF) transfer and refine multiscale structural features.

The generator is supervised by two complementary discriminators: a U-Net-based CNN discriminator for local edges, textures, and intensity transitions, and an Intensity-Statistics-Conditioned Mamba Discriminator (ISC-MD) that uses image-level infrared intensity statistics to condition long-range feature modeling.

## Highlights

- A two-stage framework for remote-sensing visible-to-infrared image translation.
- Cross-modal structural-prior learning through masked reconstruction and contrastive representation alignment.
- Transfer of the pretrained visible encoder to provide a structure-aware initialization for infrared image generation.
- Cross-Layer Skip Connections (CL-Skip) for transferring multiscale structural information between the encoder and decoder.
- Cross-level Selective Attention Fusion (CSAF) for adaptively integrating adjacent shallow and deep features.
- A dual-discriminator design combining local convolutional supervision with image-wide long-range adversarial assessment.
- An Intensity-Statistics-Conditioned Mamba Discriminator (ISC-MD) that encodes the mean, standard deviation, and soft histogram of an infrared image into a conditioning token.
- Evaluation on FMB, VEDAI, and AVIID-1 using structural, pixel-level, perceptual, statistical, spatial, and frequency-domain measures.
- Downstream object-detection experiments for assessing the practical utility of generated infrared images.

## Framework

StruIR consists of two training stages.

### Stage 1: Cross-Modal Structural Prior Learning

Two encoder-decoder branches process paired visible and infrared images. Masked image reconstruction encourages the encoders to recover contours, region boundaries, object layouts, and other structural information from incomplete observations.

Cross-modal contrastive learning is further applied at the encoder bottleneck to align the representations of paired visible and infrared images. The two objectives jointly encourage the visible encoder to learn structural representations that can be transferred to the subsequent generation stage.

### Stage 2: Structural-Prior-Guided Generation

The pretrained visible encoder from Stage 1 is used to initialize the encoder of the VI-to-IR generator. During Stage-2 training, the transferred encoder remains trainable and is jointly optimized with the decoder.

CL-Skip modules are introduced into the intermediate and shallow skip pathways. Each CL-Skip uses CSAF to align and selectively fuse the current-scale feature with its adjacent shallow and deep features. This allows deeper semantic information to guide the recovery of shallow boundaries and spatial details while reducing the direct transmission of visible-modality-specific textures.

The generated infrared image is evaluated by two complementary discriminators:

- A U-Net-based CNN discriminator that focuses primarily on local textures, edges, intensity transitions, and correspondence with the visible input.
- ISC-MD, which incorporates image-level intensity statistics into Mamba-based long-range feature modeling for complementary image-wide adversarial supervision.

Only the generator is required during inference.

## Repository Structure

```text
StruIR/
|-- train.py                       # Training entry point
|-- test.py                        # Image generation and qualitative testing
|-- evaluate.py                    # Quantitative evaluation
|-- eval.py                        # Validation helper used during training
|-- requirements.txt               # Python dependencies
|-- data/                          # Dataset loaders
|-- models/                        # Models and loss functions
|   |-- physmamba.py               # Main StruIR training model
|   |-- networks.py                # Generator and common network components
|   |-- CSAF.py                    # CL-Skip and CSAF modules
|   +-- unetgan/
|       |-- mamba_discriminator.py # ISC-MD implementation
|       +-- self_perceptual_loss.py
|-- options/                       # Training and testing options
|-- util/                          # Visualization and utility functions
+-- lpips/                         # LPIPS implementation and weights
```

> **Note:** Some internal filenames, class names, command-line options, and checkpoint names retain earlier identifiers such as `physmamba`, `D_mamba`, or `IDM-D` for compatibility with the released code. In the paper and this README, the Mamba-based discriminator is referred to as **ISC-MD**.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n struir python=3.8 -y
conda activate struir
pip install -r requirements.txt
```

ISC-MD also depends on Mamba-related packages. If they are not included in the environment file, install them separately:

```bash
pip install einops timm mamba-ssm pytorch-fid
```

> **Note:** `mamba-ssm` may require a CUDA-compatible PyTorch installation. Please select versions compatible with your local CUDA and PyTorch environment.

## Data Preparation

For paired visible-to-infrared training, organize the dataset as follows:

```text
datasets/
+-- VEDAI/
    |-- train/
    |   |-- 00000001_vis.png
    |   |-- 00000001_ir.png
    |   |-- 00000002_vis.png
    |   +-- 00000002_ir.png
    +-- test/
        |-- 00000003_vis.png
        |-- 00000003_ir.png
        |-- 00000004_vis.png
        +-- 00000004_ir.png
```

The visible image is treated as domain A, and the infrared image is treated as domain B. Paired samples should share the same identifier and use the `_vis` and `_ir` suffixes.

The datasets used in the paper are:

- FMB
- VEDAI
- AVIID-1

For these image-based datasets, the expected structure is:

```text
<dataroot>/train/*_vis.png
<dataroot>/train/*_ir.png
<dataroot>/test/*_vis.png
<dataroot>/test/*_ir.png
```

Additional dataset loaders, including M3FD, KAIST, LLVIP, and FLIR, are retained in the codebase.

For LLVIP, the current loader expects JPEG files:

```text
*_vis.jpg
*_ir.jpg
```

For FLIR, the loader expects preprocessed NumPy arrays:

```text
grayscale_training_data.npy
thermal_training_data.npy
grayscale_test_data.npy
thermal_test_data.npy
```

## Stage-1 Pretrained Encoder

Before Stage-2 generator training, complete the cross-modal structural-prior learning stage and prepare a checkpoint containing the pretrained visible encoder:

```text
vis_encoder
```

Example checkpoint path:

```text
pretrained/encoders_best.pth
```

Place the checkpoint under `pretrained/` or provide its path using:

```text
--pretrained_encoder_path
```

Only the visible encoder is transferred to the Stage-2 generator. The reconstruction decoders and contrastive projection heads used during Stage 1 are not transferred.

## Training

Example Stage-2 training command on VEDAI:

```bash
python train.py \
  --dataset_mode VEDAI \
  --dataroot ./datasets/VEDAI \
  --name struir_vedai \
  --model physmamba \
  --which_model_netG unet_512 \
  --which_model_netD unetdiscriminator \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --lambda_A 100 \
  --norm instance \
  --pool_size 0 \
  --loadSize 512 \
  --fineSize 512 \
  --batchSize 4 \
  --nThreads 4 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --pretrained_encoder_path ./pretrained/encoders_best.pth
```

Checkpoints and training logs are saved to:

```text
checkpoints/<experiment_name>/
```

To resume training:

```bash
python train.py \
  --dataset_mode VEDAI \
  --dataroot ./datasets/VEDAI \
  --name struir_vedai \
  --model physmamba \
  --continue_train \
  --which_epoch latest \
  --checkpoints_dir ./checkpoints \
  --pretrained_encoder_path ./pretrained/encoders_best.pth
```

A separate StruIR model should be trained for each dataset. The experiments reported in the paper follow independently trained within-dataset evaluation settings rather than a cross-dataset transfer protocol.

## Testing

Generate infrared images from visible inputs:

```bash
python test.py \
  --dataset_mode VEDAI \
  --dataroot ./datasets/VEDAI \
  --name struir_vedai \
  --model physmamba \
  --which_epoch 200 \
  --phase test \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --loadSize 512 \
  --fineSize 512 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results \
  --how_many 100000
```

Generated images are saved to:

```text
results/<experiment_name>/test_<epoch>/
```

Only the generator is used during inference. The two discriminators are required only during training.

## Evaluation

Run the quantitative evaluation:

```bash
python evaluate.py \
  --dataset_mode VEDAI \
  --dataroot ./datasets/VEDAI \
  --name struir_vedai \
  --model physmamba \
  --which_epoch 200 \
  --phase test \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --loadSize 512 \
  --fineSize 512 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results
```

The evaluation code supports commonly used generation-quality and efficiency metrics, including:

```text
SSIM, MS-SSIM, PSNR, L1, LPIPS, FID, FPS
```

The paper additionally compares generated and real infrared images using measures related to:

- Intensity distributions
- Local intensity variations
- Spatial autocorrelation
- Frequency-domain characteristics

## Main Results

Across the 12 conventional dataset-metric comparisons on FMB, VEDAI, and AVIID-1, StruIR ranks within the top two in every case and achieves the best or tied-best result in nine comparisons.

The complementary experiments show that StruIR produces small discrepancies from real infrared images in gradient characteristics, spatial autocorrelation, and radial power spectra.

In the downstream object-detection experiment, a detector trained using StruIR-generated infrared images achieves an AP50:95 of 0.633 on real infrared test images, compared with 0.640 when trained using real infrared images under the same protocol.

These results indicate that StruIR preserves scene structures and produces infrared images with useful intensity, spatial, and frequency-domain characteristics.

## Checkpoint Naming

The current training code saves model weights using the following format:

```text
<epoch>_net_G.pth
<epoch>_net_D.pth
<epoch>_net_D_mamba.pth

latest_net_G.pth
latest_net_D.pth
latest_net_D_mamba.pth
```

For example, when testing with `--which_epoch 200`, place the following files under `checkpoints/<experiment_name>/`:

```text
200_net_G.pth
200_net_D.pth
200_net_D_mamba.pth
```

Here, `D_mamba` corresponds to ISC-MD. The earlier checkpoint identifier is retained for compatibility.

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{sun2026struir,
  title   = {StruIR: Structural Prior Transfer with an Intensity-Statistics-Conditioned Mamba Discriminator for Visible-to-Infrared Image Translation},
  author  = {Sun, Xiaokun and Li, Shuang and Hu, Canbin},
  journal = {Under review},
  year    = {2026}
}
```

The BibTeX entry will be updated after publication.

## Acknowledgements

This codebase is developed based on the PyTorch implementations of pix2pix/CycleGAN and InfraGAN, with additional components for cross-modal structural-prior learning, structural-prior transfer, CL-Skip feature fusion, and intensity-statistics-conditioned long-range adversarial modeling.

We thank the authors of the related open-source projects for making their code publicly available.

## License

This project is released under the MIT License.