import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
from nvae.vae_celeba import NVAE


class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        预热KL损失，先对各级别的KL损失进行预热，预热完成后，对M_N的值进行衰减,所有衰减策略采用线性衰减
        :param init_weights: 各级别 KL 损失的初始权重
        :param steps: 各级别KL损失从初始权重增加到1所需的步数
        :param M_N: 初始M_N值
        :param eta_M_N: 最小M_N值
        :param M_N_decay_step: 从初始M_N值到最小M_N值所需的衰减步数
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            # 如果所有级别的KL损失的预热都已完成
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                # 准备M_N的衰减
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss


def train(opt):
    # DDP
    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    epochs = opt.epochs
    batch_size = opt.batch_size

    # device = "cuda:"+str(local_rank)

    model = NVAE(z_dim=512, img_dim=(opt.img_sz, opt.img_sz)).to(local_rank)

    # apply Spectral Normalization
    model.apply(add_sn)

    if dist.get_rank() == 0 and opt.pretrained_weights:
        print('use pretrained weights')
        model.load_state_dict(torch.load(opt.pretrained_weights), strict=False)

    # DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # train/0
    # dataset_path = [os.path.join(opt.dataset_path, '0'), os.path.join(opt.dataset_path, '1')]
    dataset_path = [opt.dataset_path]

    train_ds = ImageFolderDataset(dataset_path, img_dim=opt.img_sz)
    # print('dataset_num:' + str(len(train_ds)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=opt.n_cpu, sampler=train_sampler)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                             steps=[4500, 3000, 1500],
                             M_N=opt.batch_size / len(train_ds),
                             eta_M_N=5e-6,
                             M_N_decay_step=36000)
    # print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    step = 0
    min_loss = 1e10
    for epoch in range(epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        n_true = 0
        total_size = 0
        total_loss = 0
        for i, image in enumerate(train_dataloader):
            optimizer.zero_grad()

            image = image.to(local_rank)
            image_recon, recon_loss, kl_losses = model(image)
            kl_loss = warmup_kl.get_loss(step, kl_losses)
            loss = recon_loss + kl_loss

            print("\r---- [Local Rank %d, Epoch %d/%d, Step %d/%d] loss: %.6f----" % (local_rank, epoch, epochs, i, len(train_dataloader), loss.item()))

            total_loss = loss.item()
            loss.backward()
            optimizer.step()

            step += 1

        scheduler.step()

        if dist.get_rank() == 0:
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), f"checkpoints/IMDB-WIKI_160_DDP/%d_%.6f.pth" % (epoch, total_loss))

            if epoch % 25 == 0:
                torch.save(model.state_dict(), f"checkpoints/IMDB-WIKI_160_DDP/%d_%.6f.pth" % (epoch, total_loss))

            model.eval()

            with torch.no_grad():
                z = torch.randn((1, 512, 2, 2)).to(local_rank)
                gen_img, _ = model.module.decoder(z)
                gen_img = gen_img.permute(0, 2, 3, 1)
                gen_img = gen_img[0].cpu().numpy() * 255
                gen_img = gen_img.astype(np.uint8)

                plt.title(str(epoch))
                plt.imshow(gen_img)
                plt.savefig(f"output/DDP/%d_%.6f.png" % (epoch, total_loss))
                plt.show()
                plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trainer for state AutoEncoder model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each sample batch")
    # parser.add_argument("--dataset_path", type=str, default='data/IMDB-WIKI', help="dataset path")
    parser.add_argument("--dataset_path", type=str, default='data/IMDB-WIKI/', help="dataset path")
    parser.add_argument("--pretrained_weights", type=str, default='', help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument('--img_sz', type=int, default=160, help='image size')
    parser.add_argument("--local_rank", default=-1, type=int)
    opt = parser.parse_args()

    train(opt)
