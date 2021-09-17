# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 19:34
# @Author  : Wu Weiran
# @File    : train.py
import os
import torch
import config
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt
from utils import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader
from dataset import HorseZebraDataset
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.utils import save_image


def lambda_decay(epoch):
    warm_epoch = 100
    if epoch <= warm_epoch:
        return 1
    else:
        return 1 + (warm_epoch - epoch) / (config.num_epoch - warm_epoch)


def denorm(x):
    return (x + 1) / 2


def train_fn(disc_H, disc_Z, gen_H, gen_Z, data_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(data_loader, leave=True)  # leave参数用于迭代完成后是否保留进度条

    for i, (horse, zebra) in enumerate(loop):
        horse = horse.to(config.device)
        zebra = zebra.to(config.device)

        # Train the Discriminator H and Z
        with torch.cuda.amp.autocast():  # 自动混合精度训练
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_H(zebra)
            D_Z_fake = disc_H(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it together
            D_loss = (D_H_loss + D_Z_loss) / 2  # 除以2是为了降低判别器相对于生成器的学习速度

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()  # 放大梯度
        # 首先把梯度的值unscale回来.
        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        d_scaler.step(opt_disc)
        d_scaler.update()  # dynamically updates the scale for next iteration.

        # Train the Generator H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generator
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            G_H_loss = mse(D_H_fake, torch.ones_like(D_H_fake))
            G_Z_loss = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_horse_loss = l1(cycle_horse, horse)
            cycle_zebra_loss = l1(cycle_zebra, zebra)

            # add all together
            G_loss = G_H_loss + G_Z_loss + config.lambda_cycle * (cycle_zebra_loss + cycle_horse_loss)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if (i % 200) == 0:
            if not os.path.exists("samples/epoch{}".format(epoch)):
                os.makedirs("samples/epoch{}/horse".format(epoch))
                os.makedirs("samples/epoch{}/zebra".format(epoch))
            save_image(denorm(fake_horse), "samples/epoch{}/horse/{}.png".format(epoch, i))
            save_image(denorm(fake_zebra), "samples/epoch{}/zebra/{}.png".format(epoch, i))
        loop.set_postfix(H_real=H_reals / (i + 1), H_fake=H_fakes / (i + 1))  # 设置进度条文字


def main():
    disc_H = Discriminator().to(config.device)
    disc_Z = Discriminator().to(config.device)
    gen_H = Generator(img_channels=3, num_residual=9).to(config.device)
    gen_Z = Generator(img_channels=3, num_residual=9).to(config.device)
    opt_disc = torch.optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    opt_gen = torch.optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )
    sch_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, lambda_decay)
    sch_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lambda_decay)

    l1 = nn.L1Loss()  # 用于循环一致性损失
    mse = nn.MSELoss()  # 用于最小二乘形式的对抗损失

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_H, opt_gen, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_DISC_H, disc_H, opt_disc, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, config.learning_rate)

    dataset = HorseZebraDataset(
        root_horse=config.train_dir + "/horse",
        root_zebra=config.train_dir + "/zebra",
        transform=config.transforms
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True  # 当内存充足时，将数据加载到锁页内存中，可以加速与GPU的转换
    )

    # 在训练最开始之前实例化GradScaler对象
    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    lr_list = []
    for epoch in range(config.num_epoch):
        train_fn(disc_H, disc_Z, gen_H, gen_Z, data_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch)
        sch_disc.step()  # 更新学习率
        sch_gen.step()
        lr_list.append(opt_gen.state_dict()['param_groups'][0]['lr'])

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z)
    plt.plot(range(config.num_epoch), lr_list, color='r')
    plt.show()


if __name__ == '__main__':
    main()
