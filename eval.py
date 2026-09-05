from ssim import SSIM
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import torch.utils.data

class Evalulate:
    def __init__(self, opt):
        mode="test"
        if opt.dataset_mode == 'VEDAI':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'KAIST':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode=mode)
        elif opt.dataset_mode == 'FMB':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'FLIR':
            dataset = FlirDataset()
            dataset.initialize(opt, test=True)
        elif opt.dataset_mode == 'LLVIP':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'AVIID_1':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'M3FD':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'DRTV30K':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opt.nThreads))

    def eval(self,  model, visualizer):
        ssim_obj = SSIM(size_average=False, val_range=1.0)
        ssim_sum = 0.0
        image_count = 0
        for i, data in enumerate(self.dataloader):
            model.set_input(data)
            model.test()
            visualizer.add_errors(model.get_current_errors())
            real_b_ssim = (model.real_B + 1.0) / 2.0
            fake_b_ssim = (model.fake_B + 1.0) / 2.0
            per_image_ssim = ssim_obj(real_b_ssim, fake_b_ssim)
            ssim_sum += per_image_ssim.sum().item()
            image_count += per_image_ssim.numel()
        if image_count == 0:
            raise RuntimeError('The validation dataset is empty')
        visualizer.append_error_hist(i, val=True)

        return ssim_sum / image_count
