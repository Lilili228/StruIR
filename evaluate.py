import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
from options.test_options import TestOptions
from models.models import create_model
from pytorch_fid import fid_score
from ssim import SSIM
from lpips.lpips import LPIPS
from util import util
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import tensorflow as tf


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



vedai_class = {1: "car", 2: "truck", 11: "pickup", 4: "tractor", 5: "campingcar", 23: "boat", 7: "motorcycle", 8: "bus",
               9: "van", 10: "other", 31: "largevehicles"}
kaist_class = {1: "cyclist", 2: "people", 3: "person", 4: "other"}

opt = TestOptions().parse()
opt.nThreads = 0
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


fid_real_dir = os.path.join(opt.results_dir, 'fid_real_temp')
fid_fake_dir = os.path.join(opt.results_dir, 'fid_fake_temp')
for d in [fid_real_dir, fid_fake_dir]:
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


if opt.dataset_mode == 'FLIR':
    dataset = FlirDataset()
    dataset.initialize(opt, test=True)
else:
    dataset = ThermalDataset()
    dataset.initialize(opt, mode="test")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# 模型初始化与模式设置
model = create_model(opt)
if hasattr(model, 'netG'):
    model.netG.eval()


ssim_obj = SSIM(size_average=False, val_range=1.0).to(device)
lpips_obj = LPIPS(
    pretrained=True,
    net='alex',
    version='0.1',
    lpips=True,
    eval_mode=True,
).to(device)
lpips_obj.eval()
L1_obj = torch.nn.L1Loss().to(device)

metrics = {}
dataset_class = vedai_class if opt.dataset_mode == 'VEDAI' else kaist_class
for k in dataset_class.values():
    metrics[k] = {m: AverageMeter() for m in ["ssim", "lpips", "l1", "psnr"]}
overall_metric = {m: AverageMeter() for m in ["ssim", "lpips", "l1", "psnr"]}


def preprocess_for_fid(tensor):

    img = tensor.detach().cpu().squeeze().numpy()

    img = ((img + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def to_zero_one(tensor):
    return (tensor + 1.0) / 2.0


def to_lpips_rgb(tensor):
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    if tensor.shape[1] != 3:
        raise ValueError('LPIPS expects one-channel or three-channel images')
    return tensor


t = 0.
print(f"Starting evaluation... Saving images to {fid_fake_dir}")

for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    model.set_input(data)
    current_time = model.test()
    t += current_time


    fake_B = model.fake_B.to(device)
    real_B = model.real_B.to(device)


    fake_img_np = preprocess_for_fid(fake_B)
    real_img_np = preprocess_for_fid(real_B)
    img_name = f"{i:06d}.png"
    util.save_image(fake_img_np, os.path.join(fid_fake_dir, img_name))
    util.save_image(real_img_np, os.path.join(fid_real_dir, img_name))


    with torch.no_grad():
        l1 = L1_obj(real_B, fake_B).item()
        ssim_values = ssim_obj(to_zero_one(real_B), to_zero_one(fake_B))
        lpips_values = lpips_obj(
            to_lpips_rgb(real_B),
            to_lpips_rgb(fake_B),
            normalize=False,
        ).reshape(real_B.shape[0], -1).mean(dim=1)
        ssim = ssim_values.mean().item()
        lpips = lpips_values.mean().item()


        rb_np = real_B.detach().cpu().numpy()
        fb_np = fake_B.detach().cpu().numpy()
        psnr = tf.image.psnr(tf.convert_to_tensor(rb_np), tf.convert_to_tensor(fb_np), 2.0).numpy()
        if isinstance(psnr, np.ndarray): psnr = np.mean(psnr)


    batch_size = model.input_A.shape[0]
    overall_metric['l1'].update(l1, batch_size)
    overall_metric['ssim'].update(ssim, batch_size)
    overall_metric['lpips'].update(lpips, batch_size)
    overall_metric['psnr'].update(psnr, batch_size)


print("\nCalculating FID...")
fid_value = -1.0
try:
    fid_value = fid_score.calculate_fid_given_paths(
        [fid_real_dir, fid_fake_dir], batch_size=50, device=device, dims=2048
    )
except Exception as e:
    print(f"FID Calculation Error: {e}")


print("\nOverall results------>")
results_str = ", ".join([f"{m}={overall_metric[m].avg:.4f}" for m in overall_metric.keys()])
print(f"{results_str}, FID={fid_value:.4f}")
print(f"FPS: {(i + 1) / t:.2f}")
