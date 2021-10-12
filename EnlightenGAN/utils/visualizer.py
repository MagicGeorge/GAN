import os
import numpy as np
import time
from . import util


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:  # "a"表示打开一个文件用于追加, 新的内容将会被写入到已有内容之后
            now = time.strftime("%c")  # 时间格式为：Tue Oct 12 21:19:35 2021
            log_file.write('================ Training Loss (%s) ================\n' % now)
