import os
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

mode = "test"
if opt.dataset_mode == 'VEDAI':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode="test")
elif opt.dataset_mode == 'FMB':
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
elif opt.dataset_mode == 'KAIST':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode=mode)
elif opt.dataset_mode == 'FLIR':
    dataset = FlirDataset()
    dataset.initialize(opt, test=True)
elif opt.dataset_mode == 'BUTR':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode=mode)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads)
)

print('dataset_mode =', opt.dataset_mode)
print('dataroot =', opt.dataroot)
print('how_many =', 'all' if opt.how_many is None else opt.how_many)
print('dataset len =', len(dataset))

model = create_model(opt)
model.netG.eval()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(
    web_dir,
    'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch)
)

save_img_dir = opt.save_img_dir or os.path.join(web_dir, 'generated')
os.makedirs(save_img_dir, exist_ok=True)

for i, data in enumerate(dataset):
    if opt.how_many is not None and i >= opt.how_many:
        break

    for key in ['A', 'B']:
        if key in data and torch.is_tensor(data[key]):
            if len(data[key].shape) == 3:  # [C, H, W]
                data[key] = data[key].unsqueeze(0)  # [1, C, H, W]

    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()

    print('%04d: process image... %s' % (i, img_path))


    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
    webpage.save()


    if isinstance(img_path, list):
        current_path = img_path[0]
    else:
        current_path = img_path


    base_name = os.path.splitext(os.path.basename(current_path))[0]


    base_name = base_name.replace('_vis', '')


    for label, image_numpy in visuals.items():
        if 'fake' in label.lower():
            save_path = os.path.join(save_img_dir, '{}_fake.png'.format(base_name))
            Image.fromarray(image_numpy).save(save_path)
            print('saved:', save_path)
            break
