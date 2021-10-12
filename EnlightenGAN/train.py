import torch
import time
from tqdm import tqdm
from options import TrainOptions
from dataset import CustomDataset
from single_model import SingleModel
from torch.utils.data import DataLoader


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters Configuration
    opt = TrainOptions().parse()
    opt.patchD = True
    opt.patch_vgg = True

    # Dataset
    custom_dataset = CustomDataset(opt.data_root)

    # DataLoader
    data_loader = DataLoader(
        dataset=custom_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = SingleModel(opt, device)

    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        for i, data in enumerate(data_loader):
            total_steps += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


if __name__ == '__main__':
    main()
