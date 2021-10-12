import random
import numpy as np
import torch


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:  # 如果池子里的图片数量没有超过池子大小
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)  # 一边往池子里放
                return_images.append(image)  # 一边直接拿来用
            else:  # 池子满了
                p = random.uniform(0, 1)
                if p > 0.5:  # 一半的概率从池子中取出一个旧的，然后放进去一个新的
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # 另一半概率新的图片就不放进池子，直接拿来用
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
