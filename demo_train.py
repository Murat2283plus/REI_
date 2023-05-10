import torch
import argparse

from rei.rei import REI

# from dataset.mridb import MRIData
# from dataset.cvdb import CVDB_CVPR
from dataset.ctdb import CTData

# from physics.mri import MRI
from physics.inpainting import Inpainting
from physics.ct import CT

from transforms.shift import Shift
from transforms.rotate import Rotate


'''
# --------------------------------------------
# training code for REI
# --------------------------------------------
# Dongdong Chen (d.chen@ed.ac.uk)
# github: https://github.com/edongdongchen/EI
#         https://github.com/edongdongchen/REI
#
# Reference:
@inproceedings{chen2021equivariant,
  title     = {Equivariant Imaging: Learning Beyond the Range Space},
  author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {4379-4388}
}
@inproceedings{chen2022robust,
  title     = {Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements},
  author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
# --------------------------------------------
'''

#-------------------------------murat
class Args:
    def __init__(self):
        '''
        self.gpu: 使用的GPU设备的索引。如果您有多个GPU，可以通过设置此参数来选择特定的GPU。
        self.schedule: 学习率调度策略的列表，表示在哪些epoch更改学习率。这里的值表示在第2000个、第3000个和第4000个epoch时更新学习率。
        self.cos: 是否使用余弦退火调度策略。如果为True，则使用余弦退火；否则，使用预设的学习率调度策略。
        self.epochs: 训练的总epoch数。
        self.lr: 初始学习率。
        self.weight_decay: L2正则化的权重衰减系数。
        self.batch_size: 训练时每个批次的样本数量。
        self.ckp_interval: 保存模型检查点的epoch间隔。
        self.resume: 从先前保存的检查点恢复训练时提供的检查点文件路径。
        self.n_trans: 变换操作（如旋转、平移等）的数量，用于数据增强。
        self.alpha_req: REQ（对应于论文中的MSE损失）损失的权重。
        self.alpha_sure: SURE（对应于论文中的无监督损失）损失的权重。
        self.alpha_eq: Eq（对应于论文中的等价性损失）损失的权重。
        self.alpha_mc: MC（对应于论文中的蒙特卡洛损失）损失的权重。
        self.tau: SURE损失的超参数。
        self.task: 选择任务类型（'ct'或'mri'），分别表示计算机断层扫描和磁共振成像任务。
        self.acceleration: MRI任务中的加速因子。
        self.ct_views: CT任务中的视图数量。
        self.ct_I0: CT任务中的光子数。
        self.mask_rate: 掩码率，表示数据中被遮盖的部分。
        self.noise_type: 噪声类型，例如高斯噪声（'g'）。
        self.noise_sigma: 高斯噪声的标准差。
        self.noise_gamma: 噪声模型的γ参数。
        '''
        self.gpu = 2
        self.schedule = [20, 50, 100] # 或其他值，取决于您的任务
        self.cos = False
        self.epochs = 300
        self.lr = 1e-1
        self.weight_decay = 1e-8
        self.batch_size = 2
        self.ckp_interval = 1
        self.resume = ''
        self.n_trans = 3
        self.alpha_req = 1.0
        self.alpha_sure = 1.0
        self.alpha_eq = 1.0
        self.alpha_mc = 1.0
        self.tau = 1e-2
        self.task = 'ct' # 或 'ct' 或 'mri'
        self.acceleration = 4
        self.ct_views = 50
        self.ct_I0 = 1e5
        self.mask_rate = 0.3
        self.noise_type = 'g'
        self.noise_sigma = 0.1
        self.noise_gamma = 0.1

import os

def get_pretrained_path():
    ct_path = './ct.py'
    pretrained = None

    if os.path.exists(ct_path):
        pretrained = './ct.pt'

    return pretrained



def main():
    # args = parser.parse_args()
    args = Args() #murat
    # device=f'cuda:{args.gpu}' #murat
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') # 检测是否有GPU可用
    # pretrained = None#murat
    pretrained = get_pretrained_path()
    lr_cos = False #murat
    save_ckp = True #murat
    report_psnr = True #murat
    if args.task == 'ct':
        n_views = 50 # number of views
        tau = 10 # SURE

        epochs = 300
        ckp_interval = 2
        schedule = [100, 200]

        batch_size = 2
        lr = {'G': 5e-4, 'WD': 1e-8}
        alpha = {'req': 1e3, 'sure': 1e-5}

        # define a MPG noise model
        I0 = 1e5
        noise_sigam = 30
        noise_model = {'noise_type': 'mpg', # mixed poisson-gaussian
                       'sigma': noise_sigam,
                       'gamma': 1}

        dataloader = torch.utils.data.DataLoader(
            dataset=CTData(mode='train'), batch_size=batch_size, shuffle=True)


        transform = Rotate(n_trans=2, random_rotate=True)

        physics = CT(256, n_views, circle=False, device=device, I0=I0,
                     noise_model=noise_model)

        rei = REI(in_channels=1, out_channels=1, img_width=256, img_height=256,
                  dtype=torch.float, device=device)

        rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                      schedule, pretrained, lr_cos, save_ckp, tau, report_psnr, args)


if __name__ == '__main__':
    main()