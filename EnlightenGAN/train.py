import torch
import time
from tqdm import tqdm
from options import TrainOptions
from dataset import CustomDataset
from torch.utils.data import DataLoader


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters Configuration
    opt = TrainOptions().parse()

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

    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        loop = tqdm(data_loader, leave=True)
        for i, (image, norm_image, gray) in enumerate(loop):
            image = image.to(device)
            # gray = gray.to(device)
            print(image.shape)


if __name__ == '__main__':
    main()
