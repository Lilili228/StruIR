# https://github.com/jorge-pessoa/pytorch-msssim
import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma, device=None, dtype=None):
    dtype = dtype or torch.float32
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        device=device,
        dtype=dtype,
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1, device=None, dtype=None):
    _1D_window = gaussian(window_size, 1.5, device=device, dtype=dtype).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1.0):
    if img1.shape != img2.shape:
        raise ValueError('SSIM inputs must have identical shapes')
    if img1.ndim != 4:
        raise ValueError('SSIM inputs must be 4D NCHW tensors')
    if val_range <= 0:
        raise ValueError('val_range must be positive')

    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(
            real_size,
            channel=channel,
            device=img1.device,
            dtype=img1.dtype,
        )
    else:
        window = window.to(device=img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs_per_image = (v1 / v2).flatten(1).mean(dim=1)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    per_image = ssim_map.flatten(1).mean(dim=1)
    ret = per_image.mean() if size_average else per_image
    cs = cs_per_image.mean() if size_average else cs_per_image

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=1.0, normalize="relu"):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.register_buffer('window', create_window(window_size))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if (
            channel == self.channel
            and self.window.dtype == img1.dtype
            and self.window.device == img1.device
        ):
            window = self.window
        else:
            window = create_window(
                self.window_size,
                channel,
                device=img1.device,
                dtype=img1.dtype,
            )
            self.window = window
            self.channel = channel

        return ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
            val_range=self.val_range,
        )


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

if __name__=='__main__':
    x=torch.rand(4,3,512,512)
    y=torch.rand(4,3,512,512)

    sobj=SSIM()
    t=sobj(x,y)
    l=0
    for i in range(4):
        l+=sobj(x[i].unsqueeze(0),y[i].unsqueeze(0))

    l/=4
    l*=1
