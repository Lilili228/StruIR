import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


INSTANCE_NORM = functools.partial(nn.InstanceNorm2d, affine=True)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=INSTANCE_NORM):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = norm_layer(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=INSTANCE_NORM, use_dropout=False):
        super(UpBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.norm(self.deconv(x)))
        if self.use_dropout:
            x = self.dropout(x)
        return x


class UnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=INSTANCE_NORM):
        super(UnetEncoder, self).__init__()

        self.initial = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)

        self.down1 = DownBlock(ngf, ngf * 2, norm_layer)
        self.down2 = DownBlock(ngf * 2, ngf * 4, norm_layer)
        self.down3 = DownBlock(ngf * 4, ngf * 8, norm_layer)
        self.down4 = DownBlock(ngf * 8, ngf * 8, norm_layer)
        self.down5 = DownBlock(ngf * 8, ngf * 8, norm_layer)
        self.down6 = DownBlock(ngf * 8, ngf * 8, norm_layer)
        self.down7 = DownBlock(ngf * 8, ngf * 8, norm_layer)

        self.bottleneck_conv = nn.Conv2d(
            ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bottleneck_relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        skips = []
        d1 = self.initial(x)
        skips.append(d1)
        d2 = self.down1(d1)
        skips.append(d2)
        d3 = self.down2(d2)
        skips.append(d3)
        d4 = self.down3(d3)
        skips.append(d4)
        d5 = self.down4(d4)
        skips.append(d5)
        d6 = self.down5(d5)
        skips.append(d6)
        d7 = self.down6(d6)
        skips.append(d7)
        d8 = self.down7(d7)
        skips.append(d8)

        bottleneck = self.bottleneck_conv(self.bottleneck_relu(d8))
        return bottleneck, skips


class UnetDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, norm_layer=INSTANCE_NORM):
        super(UnetDecoder, self).__init__()

        self.up1 = UpBlock(ngf * 8, ngf * 8, norm_layer, use_dropout=True)
        self.up2 = UpBlock(ngf * 8, ngf * 8, norm_layer, use_dropout=True)
        self.up3 = UpBlock(ngf * 8, ngf * 8, norm_layer, use_dropout=True)
        self.up4 = UpBlock(ngf * 8, ngf * 8, norm_layer, use_dropout=False)
        self.up5 = UpBlock(ngf * 8, ngf * 8, norm_layer)
        self.up6 = UpBlock(ngf * 8, ngf * 4, norm_layer)
        self.up7 = UpBlock(ngf * 4, ngf * 2, norm_layer)
        self.up8 = UpBlock(ngf * 2, ngf, norm_layer)
        self.final = nn.ConvTranspose2d(
            ngf, output_nc, kernel_size=4, stride=2, padding=1
        )
        self.final_tanh = nn.Tanh()

    def forward(self, bottleneck, skips):
        u1 = self.up1(bottleneck) + skips[-1]
        u2 = self.up2(u1) + skips[-2]
        u3 = self.up3(u2) + skips[-3]
        u4 = self.up4(u3) + skips[-4]
        u5 = self.up5(u4) 
        u6 = self.up6(u5)
        u7 = self.up7(u6)
        u8 = self.up8(u7)

        return self.final_tanh(self.final(u8))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.global_pool(x)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


class DualModalReconstructionNet(nn.Module):
    def __init__(self, vis_nc=3, ir_nc=1, ngf=64, mask_ratio=0.5, patch_size=16):
        super(DualModalReconstructionNet, self).__init__()
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError('mask_ratio must be in [0, 1]')
        if patch_size <= 0:
            raise ValueError('patch_size must be positive')
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        self.vis_encoder = UnetEncoder(input_nc=vis_nc, ngf=ngf)
        self.vis_decoder = UnetDecoder(output_nc=vis_nc, ngf=ngf)
        self.ir_encoder = UnetEncoder(input_nc=ir_nc, ngf=ngf)
        self.ir_decoder = UnetDecoder(output_nc=ir_nc, ngf=ngf)
        self.projector = ProjectionHead(in_dim=ngf * 8, out_dim=128)

    def sample_mask(self, x):
        _, _, height, width = x.shape
        patch_size = self.patch_size
        patch_rows, patch_cols = height // patch_size, width // patch_size
        if patch_rows * patch_size != height or patch_cols * patch_size != width:
            raise ValueError(
                'Input size (%d, %d) must be divisible by patch_size=%d' %
                (height, width, patch_size)
            )

        if self.mask_ratio <= 0:
            return torch.zeros_like(x[:, :1, :, :])

        batch_size = x.shape[0]
        num_patches = patch_rows * patch_cols
        num_masked = min(
            num_patches,
            max(1, int(num_patches * self.mask_ratio)),
        )
        mask = torch.zeros(
            batch_size,
            num_patches,
            device=x.device,
            dtype=x.dtype,
        )
        scores = torch.rand(batch_size, num_patches, device=x.device)
        mask_indices = scores.topk(num_masked, dim=1).indices
        mask.scatter_(1, mask_indices, 1)
        return F.interpolate(
            mask.view(batch_size, 1, patch_rows, patch_cols),
            scale_factor=patch_size,
            mode='nearest',
        )

    def mask_image(self, x, mask=None):
        if mask is None:
            mask = self.sample_mask(x)
        if x.shape[0] != mask.shape[0] or x.shape[-2:] != mask.shape[-2:]:
            raise ValueError('Image and mask batch/spatial dimensions must match')
        return x * (1 - mask), mask

    def forward(self, vis_img, ir_img):
        if vis_img.shape[0] != ir_img.shape[0] or vis_img.shape[-2:] != ir_img.shape[-2:]:
            raise ValueError(
                'Paired visible and infrared images must share batch/spatial dimensions'
            )

        masked_vis, shared_mask = self.mask_image(vis_img)
        masked_ir, _ = self.mask_image(ir_img, shared_mask)

        vis_bottleneck, vis_skips = self.vis_encoder(masked_vis)
        ir_bottleneck, ir_skips = self.ir_encoder(masked_ir)

        vis_proj = self.projector(vis_bottleneck)
        ir_proj = self.projector(ir_bottleneck)

        vis_recon = self.vis_decoder(vis_bottleneck, vis_skips)
        ir_recon = self.ir_decoder(ir_bottleneck, ir_skips)

        return (
            vis_recon,
            ir_recon,
            vis_proj,
            ir_proj,
            shared_mask,
            shared_mask,
            masked_vis,
            masked_ir,
        )


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z.float(), z.float().T) / self.temperature
        labels = torch.cat([
            torch.arange(batch_size, dtype=torch.long, device=z.device) + batch_size,
            torch.arange(batch_size, dtype=torch.long, device=z.device)
        ], dim=0)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, torch.finfo(sim_matrix.dtype).min)
        return self.criterion(sim_matrix, labels)
