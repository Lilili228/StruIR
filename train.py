import time
import math
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
import torch
import os
from eval import Evalulate

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)



if (
    not opt.continue_train
    and hasattr(opt, 'pretrained_encoder_path')
    and opt.pretrained_encoder_path is not None
):
    print('Loading pretrained encoder from:', opt.pretrained_encoder_path)

    if not os.path.isfile(opt.pretrained_encoder_path):
        raise FileNotFoundError(
            'Pretrained encoder not found: %s' % opt.pretrained_encoder_path
        )

    ckpt = torch.load(opt.pretrained_encoder_path, map_location='cpu')


    encoder_state_dict = ckpt['vis_encoder']

    model.netG.encoder.load_state_dict(encoder_state_dict, strict=False)
    print('Pretrained VIS encoder loaded successfully!')


    for param in model.netG.encoder.parameters():
        param.requires_grad = True

    print('Encoder unfrozen for fine-tuning.')
elif opt.continue_train:
    print('Continue training: keeping the encoder weights from the checkpoint.')

visualizer = Visualizer(opt)
total_steps = 0
hist_error = model.get_errors()
visualizer.plot_data = {'train': OrderedDict((k, []) for k in hist_error.keys()),
                        'val': OrderedDict((k, []) for k in hist_error.keys()),
                        'legend': list(hist_error.keys())}
eval = Evalulate(opt)
best_checkpoint_path = os.path.join(model.save_dir, "best_checkpoint.pth")
best_g_loss = float("inf")
if opt.continue_train:
    p = os.path.join(model.save_dir, "history.pth")
    hist = torch.load(p)
    visualizer.plot_data = hist['plot_data']
    visualizer.metric_data = hist['metric']
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        best_g_loss = float(best_checkpoint['loss_G'])
        print('Loaded best generator loss %.6f from epoch %d' %
              (best_g_loss, best_checkpoint['epoch']))

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    epoch_g_loss_sum = 0.0
    epoch_sample_count = 0
    visualizer.data_error = [0 for _ in hist_error.keys()]
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        batch_size = model.real_A.size(0)
        epoch_g_loss_sum += float(model.loss_G_val.item()) * batch_size
        epoch_sample_count += batch_size

        # if total_steps % opt.display_freq == 0:
            # save_result = total_steps % opt.update_html_freq == 0
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        errors = model.get_current_errors()
        visualizer.add_errors(errors)
        if total_steps % opt.print_freq == 0:
            # errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
            # if opt.display_id > 0:
            #    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        iter_data_time = time.time()
    del data
    if epoch_sample_count == 0:
        raise RuntimeError('No training samples were processed in epoch %d' % epoch)
    epoch_g_loss = epoch_g_loss_sum / epoch_sample_count
    print('Epoch %d average generator loss: %.6f' % (epoch, epoch_g_loss))
    if math.isfinite(epoch_g_loss) and epoch_g_loss < best_g_loss:
        best_g_loss = epoch_g_loss
        model.save('best')
        torch.save(
            {'epoch': epoch, 'loss_G': best_g_loss},
            best_checkpoint_path,
        )
        print('Best model updated at epoch %d with generator loss %.6f' %
              (epoch, best_g_loss))
    if opt.display_id > 0:
        visualizer.append_error_hist(i)
        visualizer.data_error = [0 for _ in hist_error.keys()]
        visualizer.display_current_results(model.get_current_visuals(), epoch, True)
        ssim = eval.eval(model, visualizer)
        visualizer.plot_current_errors()
        visualizer.plot_current_metrics(ssim)
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        hist = {'plot_data': visualizer.plot_data,
                'metric': visualizer.metric_data}
        p = os.path.join(model.save_dir, "history.pth")
        torch.save(hist, p)
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
