import argparse
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from masking_vir2ir import DualModalReconstructionNet, NTXentLoss


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def make_thermal_dataset_fmb(path):
    if not os.path.isdir(path):
        raise RuntimeError('%s is not a valid directory' % path)

    images = []
    for fname in sorted(os.listdir(path)):
        if not (is_image_file(fname) and fname.endswith('_vis.png')):
            continue
        visible_path = os.path.join(path, fname)
        infrared_path = os.path.join(path, fname[:-8] + '_ir.png')
        if os.path.isfile(infrared_path):
            images.append({'A': visible_path, 'B': infrared_path})

    if not images:
        raise RuntimeError('No FMB pairs found in %s' % path)
    return images


def resolve_fmb_split_dir(data_root, split):
    """Resolve either ``root/<split>`` or a flat directory of FMB pairs."""
    data_root = os.path.abspath(os.path.expanduser(data_root))
    split_dir = os.path.join(data_root, split)

    # Keep the usual FMB/{train,val,test} layout when it exists.  The
    # FMB-pretrain dump supplied with this project is flat, so accept it
    # directly when the requested split directory is absent.
    if os.path.isdir(split_dir):
        return split_dir
    if os.path.basename(os.path.normpath(data_root)) == split:
        return data_root
    if os.path.isdir(data_root):
        has_visible_pairs = any(
            is_image_file(name) and name.endswith('_vis.png')
            for name in os.listdir(data_root)
        )
        if has_visible_pairs:
            return data_root
    return split_dir


class FMBDataset(Dataset):
    def __init__(self, root_dir, load_size=512):
        self.pairs = make_thermal_dataset_fmb(root_dir)
        self.load_size = load_size

    def __getitem__(self, index):
        pair = self.pairs[index]

        visible = Image.open(pair['A']).convert('RGB')
        visible = visible.resize((self.load_size, self.load_size), Image.BICUBIC)
        visible = transforms.ToTensor()(visible.copy())
        visible = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(visible)

        infrared = Image.open(pair['B'])
        infrared = ImageOps.grayscale(infrared)
        infrared = infrared.resize((self.load_size, self.load_size), Image.BICUBIC)
        infrared = transforms.ToTensor()(infrared.copy()).float()
        infrared = transforms.Normalize((0.5,), (0.5,))(infrared)

        return {
            'A': visible,
            'B': infrared,
            'A_paths': pair['A'],
            'B_paths': pair['B'],
        }

    def __len__(self):
        return len(self.pairs)


def denorm(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_sample_images(real_vis, real_ir, masked_vis, masked_ir,
                       recon_vis, recon_ir, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    vis_row = torch.cat([
        real_vis[0:1].detach().cpu(),
        masked_vis[0:1].detach().cpu(),
        recon_vis[0:1].detach().cpu(),
    ], dim=3)
    ir_row = torch.cat([
        real_ir[0:1].detach().cpu(),
        masked_ir[0:1].detach().cpu(),
        recon_ir[0:1].detach().cpu(),
    ], dim=3).repeat(1, 3, 1, 1)
    final_image = torch.cat([vis_row, ir_row], dim=2)
    save_image(denorm(final_image), os.path.join(save_dir, 'epoch_%04d.png' % epoch))


def make_encoder_checkpoint(model, epoch, metrics, args):
    return {
        'epoch': epoch,
        'vis_encoder': model.vis_encoder.state_dict(),
        'ir_encoder': model.ir_encoder.state_dict(),
        'metrics': metrics,
        'config': vars(args),
    }


def autocast_context(enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, 'amp'):
        return torch.amp.autocast(device_type='cuda')
    return torch.cuda.amp.autocast()


def make_grad_scaler(enabled):
    if not enabled:
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda')
    return torch.cuda.amp.GradScaler()


def parse_args():
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='FMB dual-modal MIM pretraining')
    parser.add_argument('--data-root', default='/home/lishuang/ls/datasets/FMB')
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    parser.add_argument('--save-dir', default=os.path.join(project_root, 'checkpoints_FMB'))
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--load-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-frequency', type=int, default=5)
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='CUDA device index; use -1 for CPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true',
                        help='use CUDA automatic mixed precision')
    parser.add_argument('--max-batches', type=int, default=0,
                        help='stop after this many batches (smoke test; 0 means all)')
    return parser.parse_args()


def train_fmb(args):
    set_random_seed(args.seed)
    if args.load_size % args.patch_size != 0:
        raise ValueError('--load-size must be divisible by --patch-size')

    if args.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA is unavailable. Use --gpu-id -1 only for a small CPU smoke test.'
            )
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError('CUDA device %d is not available' % args.gpu_id)
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    split_dir = resolve_fmb_split_dir(args.data_root, args.split)
    dataset = FMBDataset(root_dir=split_dir, load_size=args.load_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    image_dir = args.image_dir or os.path.join(args.save_dir, 'samples')
    model = DualModalReconstructionNet(
        vis_nc=3,
        ir_nc=1,
        ngf=args.ngf,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    reconstruction_criterion = nn.L1Loss()
    contrastive_criterion = NTXentLoss(temperature=0.1).to(device)
    amp_enabled = args.amp and device.type == 'cuda'
    scaler = make_grad_scaler(amp_enabled)

    print('Using device:', device)
    print('Dataset:', split_dir, 'pairs:', len(dataset))
    print('Saving checkpoints to:', args.save_dir)

    best_total = float('inf')
    last_metrics = None
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        sums = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'vis_l1': 0.0,
            'ir_l1': 0.0,
            'contrastive': 0.0,
        }
        batches = 0
        loop = tqdm(loader, desc='Epoch [%d/%d]' % (epoch, args.num_epochs), leave=True)

        for data in loop:
            if args.max_batches and batches >= args.max_batches:
                break
            real_vis = data['A'].to(device, non_blocking=True)
            real_ir = data['B'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                outputs = model(real_vis, real_ir)
                vis_recon, ir_recon, vis_proj, ir_proj = outputs[:4]
                loss_vis = reconstruction_criterion(vis_recon, real_vis)
                loss_ir = reconstruction_criterion(ir_recon, real_ir)
                loss_reconstruction = loss_vis + loss_ir
                loss_con = contrastive_criterion(vis_proj, ir_proj)
                total_loss = 10.0 * loss_reconstruction + 1.0 * loss_con

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            values = {
                'total_loss': total_loss.item(),
                'reconstruction': loss_reconstruction.item(),
                'vis_l1': loss_vis.item(),
                'ir_l1': loss_ir.item(),
                'contrastive': loss_con.item(),
            }
            for key, value in values.items():
                sums[key] += value
            batches += 1
            loop.set_postfix(
                Total=values['total_loss'],
                VisL1=values['vis_l1'],
                IRL1=values['ir_l1'],
                Con=values['contrastive'],
            )

        if batches == 0:
            raise RuntimeError('No batches were processed')
        metrics = {key: value / batches for key, value in sums.items()}
        last_metrics = metrics

        if epoch % args.save_frequency == 0 or epoch == args.num_epochs:
            checkpoint = make_encoder_checkpoint(model, epoch, metrics, args)
            torch.save(checkpoint, os.path.join(args.save_dir, 'encoders_epoch_%d.pth' % epoch))
            if metrics['total_loss'] < best_total:
                best_total = metrics['total_loss']
                torch.save(checkpoint, os.path.join(args.save_dir, 'encoders_best.pth'))
            save_sample_images(
                real_vis, real_ir, outputs[6], outputs[7],
                outputs[0], outputs[1], epoch, image_dir,
            )

    final_checkpoint = make_encoder_checkpoint(model, args.num_epochs, last_metrics, args)
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'encoders_final.pth'))
    print('Final encoder checkpoint:', os.path.join(args.save_dir, 'encoders_final.pth'))
    print('Best encoder checkpoint:', os.path.join(args.save_dir, 'encoders_best.pth'))


if __name__ == '__main__':
    train_fmb(parse_args())
