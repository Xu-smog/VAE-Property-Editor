# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import pandas as pd

from nvae.dataset import ImageAttrDataset
from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
from nvae.utils import reparameterize


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, opt.img_sz, interpolation=cv2.INTER_LINEAR)
    image = image / 255.
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = torch.unsqueeze(image, dim=0)
    return image


def save_image(model, folder, image_file_name, z):
    if not os.path.exists(folder):
        os.mkdir(folder)

    # model = model.cpu()
    # z = z.cpu()

    gen_img, _ = model.decoder(z)
    # print(gen_img.shape)
    gen_img = gen_img.permute(0, 2, 3, 1)
    gen_img = gen_img[0].cpu().numpy() * 255
    img = Image.fromarray(np.uint8(gen_img))

    img.save(os.path.join(folder, image_file_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default='')
    parser.add_argument("--pretrained_weights", type=str, default='')
    parser.add_argument('--img_sz', type=int, default=64, help='image size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    opt = parser.parse_args()

    device = "cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu"

    model = NVAE(z_dim=512, img_dim=(opt.img_sz, opt.img_sz))
    image = read_image(opt.image_path)
    model.apply(add_sn)
    model.to(device)
    image = image.to(device)
    model.load_state_dict(torch.load(opt.pretrained_weights, map_location=device), strict=False)
    model.eval()

    mu, log_var, xs = model.encoder(image)

    # (B, D_Z)
    z = reparameterize(mu, torch.exp(0.5 * log_var))

    with torch.no_grad():
        save_image(model, 'output', '1.jpg', z)


if __name__ == '__main__':
    main()

