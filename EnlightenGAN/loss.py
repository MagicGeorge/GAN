import os
import torch
from torch import nn
from torch import linalg
import torch.nn.functional as F
from options import TrainOptions


def load_vgg16(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    vgg = Vgg16()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))

    return vgg


def vgg_preprocess(batch, opt):  # 因为使用的是别人的预训练权重，所以模型输入的格式也必须和训练时保持一致
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939  # 为每一个batch的该通道里的所有像素值赋予同一个值
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(mean)  # subtract mean
    return batch


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        relu5_3 = F.relu(self.conv5_3(relu5_2), inplace=True)

        if opt.vgg_choose == "conv4_3":
            return relu4_3
        elif opt.vgg_choose == "conv4_2":
            return relu4_2
        elif opt.vgg_choose == "conv5_3":
            return relu5_3
        elif opt.vgg_choose == "conv5_2":
            return relu5_2
        elif opt.vgg_choose == "conv5_1":
            return relu5_1


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10

    def calc_gradient_penalty(self, netD, real_data, fake_data, opt):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates, opt)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_penalty = ((linalg.norm(gradients, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradients_penalty


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()  # Least Square GAN Loss
        else:
            self.loss = nn.BCELoss()  # vanilla GAN Loss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))  # .numel()返回数组中元素的个数
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512).cuda()
    y = torch.randn(1, 3, 512, 512).cuda()
    tanh = torch.nn.Tanh()
    x = tanh(x)
    y = tanh(y)

    opt = TrainOptions().parse()
    opt.no_vgg_instance = True
    criterionP = PerceptualLoss(opt)
    criterionWGP = DiscLossWGANGP()
    criterionVan = GANLoss()

    vgg = load_vgg16('./model')

    gp = criterionWGP.calc_gradient_penalty(vgg, x, y, opt)
    dis = criterionP.compute_vgg_loss(vgg, x, y)
    loss = criterionVan(x, True)

    print(gp)
