# StruIR

Official PyTorch implementation of **StruIR: Structural Prior Transfer and Target-Domain Distribution Regularization for Visible-to-Infrared Image Translation**.

StruIR is a two-stage visible-to-infrared image translation framework designed to improve both cross-modal structural preservation and infrared target-domain distribution consistency. The framework combines cross-modal structural-prior learning, multiscale structural transfer through CL-Skip, and distribution-aware adversarial learning with IDM-D.

## Highlights

- **Two-stage training framework** for visible-to-infrared image translation.
- **Cross-modal structural-prior learning** with masked reconstruction and contrastive learning to learn shared structural representations from paired visible and infrared images.
- **CL-Skip with Cross-level Selective Attention Fusion (CSAF)** for adjacent-scale feature fusion and structural-detail recovery.
- **Infrared Distribution-Aware Mamba Discriminator (IDM-D)** that combines image-level infrared intensity statistics with long-range spatial-context modeling.
- Evaluation support for common generation metrics, including SSIM, MS-SSIM, LPIPS, L1, PSNR, FID, and inference speed.

## Repository Structure

```text
StruIR/
|-- train.py                 # Training entry point
|-- test.py                  # Image generation / qualitative testing
|-- evaluate.py              # Quantitative evaluation
|-- eval.py                  # Validation helper used during training
|-- requirements.txt         # Python dependencies recorded for the project
|-- data/                    # Dataset loaders
|-- models/                  # StruIR generator, discriminators, and loss modules
|   |-- physmamba.py         # Main StruIR training model
|   |-- networks.py          # Generator and common network components
|   |-- CSAF.py              # CL-Skip and CSAF modules
|   +-- unetgan/
|       |-- mamba_discriminator.py  # IDM-D
|       +-- self_perceptual_loss.py # Encoder-based feature loss
|-- options/                 # Training and testing options
|-- util/                    # Visualization and utility functions
+-- lpips/                   # LPIPS implementation and weights
```

> Note: Some internal filenames, class names, command-line options, and checkpoint names retain the earlier `physmamba` or `D_mamba` identifiers for compatibility with the released code.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n struir python=3.8 -y
conda activate struir
pip install -r requirements.txt
```

IDM-D also depends on Mamba-related packages. If they are not installed by your environment file, install them separately according to your CUDA and PyTorch versions:

```bash
pip install einops timm mamba-ssm pytorch-fid
```

> Note: `mamba-ssm` may require a CUDA-compatible PyTorch build. Please install the version that matches your local CUDA and PyTorch environment.

## Data Preparation

For paired visible-to-infrared training, organize each dataset as follows:

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

The visible image is treated as domain `A`, and the infrared image is treated as domain `B`. The current dataloader supports paired samples named with `_vis` and `_ir` suffixes.

The datasets used in the paper are:

```text
FMB, VEDAI, AVIID_1
```

For these image-based datasets, use:

```text
<dataroot>/train/*_vis.png
<dataroot>/train/*_ir.png
<dataroot>/test/*_vis.png
<dataroot>/test/*_ir.png
```

Additional dataset loaders, including M3FD, KAIST, LLVIP, and FLIR, are retained in the codebase.

For `LLVIP`, the current loader expects `.jpg` files:

```text
*_vis.jpg
*_ir.jpg
```

For `FLIR`, the loader expects preprocessed NumPy arrays:

```text
grayscale_training_data.npy
thermal_training_data.npy
grayscale_test_data.npy
thermal_test_data.npy
```

## Pretrained Encoder

StruIR uses the visible encoder learned during Stage 1 to initialize the Stage-2 generator. Before training the generator, prepare a pretrained encoder checkpoint containing:

```text
vis_encoder
```

Example path:

```text
pretrained/encoders_best.pth
```

Place the pretrained weights under `pretrained/` or pass their path through `--pretrained_encoder_path`.

## Training

Example training command on VEDAI:

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

The visual results are saved to:

```text
results/<experiment_name>/test_<epoch>/
```

## Evaluation

Run quantitative evaluation:

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

The script reports:

```text
SSIM, MS-SSIM, LPIPS, L1, PSNR, FID, FPS
```

## Checkpoint Naming

The training code saves model weights using the following format:

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

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{sun2026struir,
  title   = {StruIR: Structural Prior Transfer and Target-Domain Distribution Regularization for Visible-to-Infrared Image Translation},
  author  = {Sun, Xiaokun and Li, Shuang and Hu, Canbin},
  journal = {Under review},
  year    = {2026}
}
```

The BibTeX entry will be updated after publication.

## Acknowledgements

This codebase is developed based on the PyTorch implementations of pix2pix/CycleGAN and InfraGAN, with additional modules for cross-modal structural-prior transfer, CL-Skip feature fusion, and IDM-D distribution-aware discrimination.

## License

This project is released under the [MIT License](LICENSE).