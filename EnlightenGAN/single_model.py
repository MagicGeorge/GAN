import os
import loss
import torch
import random
from utils import util
from utils.image_pool import ImagePool
import generator_model as Gen
import discriminator_model as Disc
from collections import OrderedDict


class SingleModel():
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.Tensor = torch.FloatTensor
        self.isTrain = opt.isTrain
        self.input_A = self.Tensor(opt.batch_size, 3, opt.fineSize, opt.fineSize).to(self.device)
        self.input_B = self.Tensor(opt.batch_size, 3, opt.fineSize, opt.fineSize).to(self.device)
        self.input_A_gray = self.Tensor(opt.batch_size, 1, opt.fineSize, opt.fineSize).to(self.device)
        self.image_paths = ''

        self.vgg_loss = loss.PerceptualLoss(opt)
        self.vgg_loss.to(self.device)
        self.vgg = loss.load_vgg16("./model")
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.netG_A = Gen.define_G(opt).to(self.device)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = Disc.define_D(3, opt.ndf, opt.n_layers_D, use_sigmoid).to(self.device)
            if self.opt.patchD:
                self.netD_P = Disc.define_D(3, opt.ndf, opt.n_layers_patchD, use_sigmoid).to(self.device)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.enhance_A_pool = ImagePool(opt.pool_size)

            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = loss.DiscLossWGANGP()
            else:
                self.criterionGAN = loss.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        util.print_network(self.netG_A)
        if self.isTrain:
            util.print_network(self.netD_A)
            if self.opt.patchD:
                util.print_network(self.netD_P)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_path']

    def predict(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.real_A_gray = self.input_A_gray
            if self.opt.linear_stretch:
                self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
            self.enhance_A = self.netG_A.forward(self.real_A, self.real_A_gray)

            real_A = util.tensor2im(self.real_A.data)
            enhance_A = util.tensor2im(self.enhance_A.data)
            # A_gray = util.atten2im(self.real_A_gray.data)

            return OrderedDict([('real_A', real_A), ('enhance_A', enhance_A)])

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD(real)
        pred_enhance = netD(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_enhance = pred_enhance.mean()
            loss_D = loss_D_enhance - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_enhance), True) +
                      self.criterionGAN(pred_enhance - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_enhance = self.criterionGAN(pred_enhance, False)
            loss_D = (loss_D_real + loss_D_enhance) * 0.5

        return loss_D

    def backward_D_A(self):
        enhance_A = self.enhance_A_pool.query(self.enhance_A)
        # enhance_A = self.enhance_A
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.input_B, enhance_A, True)
        self.loss_D_A.backward()

    def backward_D_P(self):
        if self.opt.only_lsgan:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.enhance_patch, False)
            if self.opt.n_patchD > 0:
                for i in range(self.opt.n_patchD):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.enhance_patch_1[i], False)
                self.loss_D_P = loss_D_P / float(self.opt.n_patchD + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.enhance_patch, True)
            if self.opt.n_patchD > 0:
                for i in range(self.opt.n_patchD):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.enhance_patch_1[i], True)
                self.loss_D_P = loss_D_P / float(self.opt.n_patchD + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P * 2
        self.loss_D_P.backward()

    def forward(self):
        self.enhance_A = self.netG_A(self.input_A, self.input_A_gray)

        if self.opt.patchD:
            w = self.input_A.size(3)
            h = self.input_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.enhance_patch = self.enhance_A[:, :, h_offset:h_offset + self.opt.patchSize,
                                 w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.input_B[:, :, h_offset:h_offset + self.opt.patchSize,
                              w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.input_A[:, :, h_offset:h_offset + self.opt.patchSize,
                               w_offset:w_offset + self.opt.patchSize]

        if self.opt.n_patchD > 0:
            self.enhance_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.input_A.size(3)
            h = self.input_A.size(2)
            for i in range(self.opt.n_patchD):
                w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.enhance_patch_1.append(self.enhance_A[:, :, h_offset:h_offset + self.opt.patchSize,
                                            w_offset:w_offset + self.opt.patchSize])
                self.real_patch_1.append(self.input_B[:, :, h_offset:h_offset + self.opt.patchSize,
                                         w_offset:w_offset + self.opt.patchSize])
                self.input_patch_1.append(self.input_A[:, :, h_offset:h_offset + self.opt.patchSize,
                                          w_offset:w_offset + self.opt.patchSize])

    def backward_G(self, epoch):
        pred_enhance = self.netD_A(self.enhance_A)
        if self.opt.use_wgan:
            loss_G_A = -pred_enhance.mean()  # 采用WGAN损失函数
        elif self.opt.use_ragan:
            pred_real = self.netD_A(self.input_B)  # 采用RaGAN损失函数
            loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_enhance), False) +
                        self.criterionGAN(pred_enhance - torch.mean(pred_real), True)) / 2
        else:
            loss_G_A = self.criterionGAN(pred_enhance, True)

        self.loss_G = loss_G_A

        loss_G_P = 0
        if self.opt.patchD:
            pred_enhance_patch = self.netD_P(self.enhance_patch)
            if self.opt.only_lsgan:  # 只用lsgan，不用ragan
                loss_G_P += self.criterionGAN(pred_enhance_patch, True)
            else:
                pred_real_patch = self.netD_P(self.real_patch)
                loss_G_P += (self.criterionGAN(pred_real_patch - torch.mean(pred_enhance_patch), False) +
                             self.criterionGAN(pred_enhance_patch - torch.mean(pred_real_patch), True)) / 2

        if self.opt.n_patchD > 0:
            for i in range(self.opt.n_patchD):
                pred_enhance_patch_1 = self.netD_P(self.enhance_patch_1[i])
                if self.opt.only_lsgan:
                    loss_G_P += self.criterionGAN(pred_enhance_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P(self.real_patch_1[i])
                    loss_G_P += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_enhance_patch_1), False) +
                                 self.criterionGAN(pred_enhance_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
            if not self.opt.D_P_times2:
                self.loss_G += loss_G_P / float(self.opt.n_patchD + 1)
            else:
                self.loss_G += loss_G_P / float(self.opt.n_patchD + 1) * 2
        else:
            if not self.opt.D_P_times2:
                self.loss_G += loss_G_P
            else:
                self.loss_G += loss_G_P * 2

        self.loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg, self.enhance_A, self.input_A)
        if self.opt.patch_vgg:
            loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, self.enhance_patch, self.input_patch)
            if self.opt.n_patchD > 0:
                for i in range(self.opt.n_patchD):
                    loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, self.enhance_patch_1[i],
                                                                     self.input_patch_1[i])
                self.loss_vgg += loss_vgg_patch / float(self.opt.n_patchD + 1)
            else:
                self.loss_vgg += loss_vgg_patch

        if epoch < 10:
            vgg_w = 0
        else:
            vgg_w = 1
        self.total_loss_G = self.loss_G + self.loss_vgg * vgg_w
        self.total_loss_G.backward()

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if self.opt.patchD == 0:
            self.optimizer_D_A.step()
        else:
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()

    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G = self.loss_G.item()
        vgg = self.loss_vgg.item()
        return OrderedDict([('D_A', D_A), ('G', G), ("vgg", vgg), ("D_P", D_P)])

    # helper saving function
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.to(self.device)

    # helper loading function
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netD_A, 'D_A', label)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
