import time
from collections import OrderedDict
import torch
from ssim import SSIM
from util import util
from . import networks
from .base_model import BaseModel


class StruIRModel(BaseModel):
    def name(self):
        return "StruIR"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and len(self.gpu_ids) > 0 else "cpu"
        )

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.which_model_netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            self.gpu_ids,
        )

        if self.isTrain:
            from .unetgan.ISCMD import ISCMD
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D,
                opt.norm,
                opt.no_lsgan,
                opt.init_type,
                self.gpu_ids,
                resolution=256 if opt.dataset_mode == "FLIR" else 512,
            )
            self.netD_iscmd = ISCMD(
                input_nc=opt.input_nc + opt.output_nc,
                embed_dim=opt.ndf * 2,
                depth=4,
                patch_size=4,
            ).to(self.device)

            if self.gpu_ids:
                self.netD_iscmd = torch.nn.DataParallel(
                    self.netD_iscmd,
                    self.gpu_ids,
                )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, "G", opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, "D", opt.which_epoch)
                self.load_network(self.netD_iscmd, "D_ISCMD", opt.which_epoch)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan,
                tensor=self.Tensor,
            )
            self.criterionL1 = torch.nn.L1Loss()
            self.ssim = SSIM()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr / 100,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D_iscmd = torch.optim.Adam(
                self.netD_iscmd.parameters(),
                lr=opt.lr / 50,
                betas=(opt.beta1, 0.999),
            )

            self.optimizers = [
                self.optimizer_G,
                self.optimizer_D,
                self.optimizer_D_iscmd,
            ]
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]

        print("---------- Networks initialized -------------")

    def set_input(self, input_data):
        a_to_b = self.opt.which_direction == "AtoB"
        input_a = input_data["A" if a_to_b else "B"]
        input_b = input_data["B" if a_to_b else "A"]

        if self.gpu_ids:
            input_a = input_a.cuda(self.gpu_ids[0], non_blocking=True)
            input_b = input_b.cuda(self.gpu_ids[0], non_blocking=True)

        self.input_A = input_a
        self.input_B = input_b
        self.image_paths = input_data["A_paths" if a_to_b else "B_paths"]

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self.netG(self.real_A)
        self.real_B = self.input_B

    def test(self):
        with torch.no_grad():
            self.real_A = self.input_A
            start_time = time.time()
            self.fake_B = self.netG(self.real_A)
            elapsed_time = time.time() - start_time
            self.real_B = self.input_B
        return elapsed_time

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_pair = torch.cat((self.real_A, self.fake_B), dim=1).detach()
        real_pair = torch.cat((self.real_A, self.real_B), dim=1)

        pred_fake_local, pred_fake_bottleneck = self.netD(fake_pair)
        pred_real_local, pred_real_bottleneck = self.netD(real_pair)

        loss_D_local_fake = self.criterionGAN(
            pred_fake_local, False
        ) + self.criterionGAN(pred_fake_bottleneck, False)
        loss_D_local_real = self.criterionGAN(
            pred_real_local, True
        ) + self.criterionGAN(pred_real_bottleneck, True)
        loss_D_local = (loss_D_local_fake + loss_D_local_real) * 0.25

        pred_fake_iscmd = self.netD_iscmd(fake_pair)
        pred_real_iscmd = self.netD_iscmd(real_pair)
        loss_D_iscmd_fake = self.criterionGAN(pred_fake_iscmd, False)
        loss_D_iscmd_real = self.criterionGAN(pred_real_iscmd, True)
        loss_D_iscmd = (loss_D_iscmd_fake + loss_D_iscmd_real) * 0.5

        self.loss_D = loss_D_local + loss_D_iscmd
        self.loss_D.backward()

        self.loss_D_real = (loss_D_local_real * 0.25 + loss_D_iscmd_real * 0.5).detach()
        self.loss_D_fake = (loss_D_local_fake * 0.25 + loss_D_iscmd_fake * 0.5).detach()
        self.loss_D_local_monitor = loss_D_local.detach()
        self.loss_D_iscmd_monitor = loss_D_iscmd.detach()

    def backward_G(self):
        fake_b_for_loss = self.fake_B.clone()
        fake_pair = torch.cat((self.real_A, fake_b_for_loss), dim=1)

        pred_fake_local, pred_fake_bottleneck = self.netD(fake_pair)
        pred_fake_iscmd = self.netD_iscmd(fake_pair)

        loss_G_GAN_local = (
            self.criterionGAN(pred_fake_local, True)
            + self.criterionGAN(pred_fake_bottleneck, True)
        ) * 0.5
        loss_G_GAN_iscmd = self.criterionGAN(pred_fake_iscmd, True)
        self.loss_G_GAN = loss_G_GAN_local * 0.8 + loss_G_GAN_iscmd * 0.2

        real_b_for_loss = self.real_B.detach()
        self.loss_G_L1 = (
            self.criterionL1(fake_b_for_loss, real_b_for_loss) * self.opt.lambda_A
        )
        fake_b_ssim = (fake_b_for_loss + 1.0) / 2.0
        real_b_ssim = (real_b_for_loss + 1.0) / 2.0
        self.loss_ssim = (
            1 - self.ssim(fake_b_ssim, real_b_ssim)
        ) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_ssim
        self.loss_G.backward()
        self.loss_G_val = self.loss_G.detach()

    def _set_discriminators_requires_grad(self, requires_grad):
        for discriminator in (self.netD, self.netD_iscmd):
            for parameter in discriminator.parameters():
                parameter.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()

        self._set_discriminators_requires_grad(True)
        self.optimizer_D.zero_grad()
        self.optimizer_D_iscmd.zero_grad()
        self.backward_D()
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self.netD_iscmd.parameters(),
            max_norm=1.0,
        )
        self.optimizer_D.step()
        self.optimizer_D_iscmd.step()

        self._set_discriminators_requires_grad(False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optimizer_G.step()
        self._set_discriminators_requires_grad(True)

    def get_current_errors(self):
        return OrderedDict(
            [
                ("G_GAN", self.loss_G_GAN),
                ("G_L1", self.loss_G_L1),
                ("D_real", self.loss_D_real),
                ("D_fake", self.loss_D_fake),
                ("D_Local", self.loss_D_local_monitor),
                ("D_ISCMD", self.loss_D_iscmd_monitor),
            ]
        )

    @staticmethod
    def get_errors():
        return OrderedDict(
            [
                ("G_GAN", 0),
                ("G_L1", 0),
                ("D_real", 0),
                ("D_fake", 0),
                ("D_Local", 0),
                ("D_ISCMD", 0),
            ]
        )

    def get_current_visuals(self):
        return OrderedDict(
            [
                ("real_A", util.tensor2im(self.real_A.detach())),
                ("fake_B", util.thermal_tensor2im(self.fake_B.detach())),
                ("real_B", util.thermal_tensor2im(self.real_B.detach())),
            ]
        )

    def save(self, label):
        self.save_network(self.netG, "G", label, self.gpu_ids)
        self.save_network(self.netD, "D", label, self.gpu_ids)
        self.save_network(self.netD_iscmd, "D_ISCMD", label, self.gpu_ids)
