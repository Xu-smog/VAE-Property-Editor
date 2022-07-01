import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from nvae.dataset import ImageAttrDataset
from nvae.utils import add_sn
from nvae.utils import reparameterize
from nvae.vae_celeba import NVAE


def compute_attribute_vector(model, image_path, image_size, attrs, data_attrs, male_attribute_vectors_file, female_attribute_vectors_file, device):

    pos_male_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    pos_female_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    neg_male_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    neg_female_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)

    pos_male_nums = torch.zeros(len(attrs), 1)
    pos_female_nums = torch.zeros(len(attrs), 1)
    neg_male_nums = torch.zeros(len(attrs), 1)
    neg_female_nums = torch.zeros(len(attrs), 1)

    # train/0
    dataset_path = [os.path.join(image_path, '0'), os.path.join(image_path, '1')]
    # dataset_path = [opt.dataset_path]

    train_ds = ImageAttrDataset(dataset_path, img_dim=image_size, attrs=data_attrs)
    print('dataset_num:' + str(len(train_ds)))
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

    for data in train_dataloader:
        image, labels = data
        image = image.to(device)

        mu, log_var, xs = model.encoder(image)
        z = reparameterize(mu, torch.exp(0.5 * log_var))
        z = z.detach().cpu()

        for i, attr in enumerate(attrs):
            if labels[attr] == 1:
                if labels['Male'] == 1:
                    pos_male_vectors[i] += z[0]
                    pos_male_nums[i] += 1
                else:
                    pos_female_vectors[i] += z[0]
                    pos_female_nums[i] += 1
            else:
                if labels['Male'] == 1:
                    neg_male_vectors[i] += z[0]
                    neg_male_nums[i] += 1
                else:
                    neg_female_vectors[i] += z[0]
                    neg_female_nums[i] += 1

    for i, num in enumerate(pos_male_nums):
            pos_male_vectors[i] /= num

    for i, num in enumerate(pos_female_nums):
            pos_female_vectors[i] /= num

    for i, num in enumerate(neg_male_nums):
            neg_male_vectors[i] /= num

    for i, num in enumerate(neg_female_nums):
            neg_female_vectors[i] /= num

    # print(pos_nums)
    # print(pos_vectors.shape)
    # print(pos_vectors)
    with torch.no_grad():
        male_attribute_vectors = {}
        female_attribute_vectors = {}

        # 测试图片
        # for i in range(len(attrs)):
        #     pos_female_images = model.decoder(pos_female_vectors[i].to(device))
        #     neg_female_images = model.decoder(neg_female_vectors[i].to(device))

        #     pos_male_images = model.decoder(pos_male_vectors[i].to(device))
        #     neg_male_images = model.decoder(neg_male_vectors[i].to(device))

        #     plot_image([img_renorm(pos_female_images[0][0].permute(1, 2, 0).cpu())],
        #                [img_renorm(neg_female_images[0][0].permute(1, 2, 0).cpu())],
        #                'female'+attrs[i])

        #     plot_image([img_renorm(pos_male_images[0][0].permute(1, 2, 0).cpu())],
        #                [img_renorm(neg_male_images[0][0].permute(1, 2, 0).cpu())],
        #                'male'+attrs[i])

        for i in range(len(attrs)):
            male_attribute_vectors[attrs[i]] = pos_male_vectors[i].cpu() - neg_male_vectors[i].cpu()
            female_attribute_vectors[attrs[i]] = pos_female_vectors[i].cpu() - neg_female_vectors[i].cpu()
            # draw the attribute for debugging
            print(attrs[i])
        
        torch.save(male_attribute_vectors, male_attribute_vectors_file)
        torch.save(female_attribute_vectors, female_attribute_vectors_file)

        return male_attribute_vectors, female_attribute_vectors


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = torch.unsqueeze(image, dim=0)
    return image


def save_image(model, folder, image_file_name, z):
    if not os.path.exists(folder):
        os.mkdir(folder)

    gen_img, _ = model.decoder(z)
    # print(gen_img.shape)
    gen_img = gen_img.permute(0, 2, 3, 1)
    gen_img = gen_img[0].cpu().numpy() * 255
    img = Image.fromarray(np.uint8(gen_img))

    img.save(os.path.join(folder, image_file_name))


def trans_attributes(model, image_path, save_path, male_attribute_vectors, female_attribute_vectors, male_attribute_vectors_file, female_attribute_vectors_file, attrs, device):
    male_attribute_vectors = torch.load(male_attribute_vectors_file)
    female_attribute_vectors = torch.load(female_attribute_vectors_file)

    model.eval()
    flag = -1
    dataset_path = [os.path.join(image_path, '0'), os.path.join(image_path, '1')]
    with torch.no_grad():
        for image_files in dataset_path:
            flag += 1
            target_path = os.path.join(save_path, image_files.split('/')[-1])
            for root, dirs, files in os.walk(image_files):
                cnt = 0
                for f in files:
                    image = read_image(os.path.join(image_files, f))
                    image = image.to(device)
                    mu, log_var, xs = model.encoder(image)
                    z = reparameterize(mu, torch.exp(0.5 * log_var))

                    z_r = z.detach()
                    for attr in attrs:
                        beta = 1.0
                        if flag == 0:
                            z_r += beta * female_attribute_vectors[attr].to(device)
                        else:
                            z_r += beta *  male_attribute_vectors[attr].to(device)

                    save_image(model, folder=target_path, image_file_name=f, z=z_r)

                    cnt += 1
                print(target_path, cnt)


def main():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()

    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--image_path", type=str, default='data/celeba/10client/train')
    parser.add_argument('--img_sz', type=int, default=64, help='image size')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--num_attrs', type=int, default=1)
    parser.add_argument("--attribute_vectors", type=str, default='checkpoints/attribute/')
    parser.add_argument('--save_path', type=str, default='data/celeba/')

    opt = parser.parse_args()

    data_attrs, attrs, attribute_vectors = None, None, None
    # 'Black_Hair', 'Smiling', 'High_Cheekbones', 'Straight_Hair'
    if opt.num_attrs == 1:
        data_attrs = ['High_Cheekbones', 'Male']
        attrs = ['High_Cheekbones']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f1-'+attrs[0])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f1-' + attrs[0])

    elif opt.num_attrs == 2:
        data_attrs = ['Black_Hair', 'Smiling', 'Male']
        attrs = ['Black_Hair', 'Smiling']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f2-'+attrs[0]+'-' +attrs[1])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f2-'+attrs[0]+'-'+attrs[1])

    elif opt.num_attrs == 3:
        data_attrs = ['Black_Hair', 'Smiling', 'High_Cheekbones', 'Male']
        attrs = ['Black_Hair', 'Smiling', 'High_Cheekbones']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f3-' + attrs[0] + '-' + attrs[1] + '-' + attrs[2])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f3-' + attrs[0] + '-' + attrs[1] + '-' + attrs[2])

    elif opt.num_attrs == 4:
        data_attrs = ['Black_Hair', 'Smiling', 'High_Cheekbones', 'Straight_Hair', 'Male']
        attrs = ['Black_Hair', 'Smiling', 'High_Cheekbones', 'Straight_Hair']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f4')
        opt.save_path = os.path.join(opt.save_path, 'VAE-f4')


    if not os.path.exists(attribute_vectors):
        os.mkdir(attribute_vectors)
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)


    device = "cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu"

    pretrained_weights = ['470_0.639811.pth', '452_0.634575.pth', '400_0.692572.pth', '475_0.705544.pth', '400_0.660960.pth', '409_0.626322.pth', '475_0.694686.pth', '403_0.642286.pth', '425_0.674690.pth', '475_0.636303.pth']


    male_attribute_vectors, female_attribute_vectors = None, None
    for i in range(opt.clients):

        model = NVAE(z_dim=512, img_dim=(opt.img_sz, opt.img_sz))

        # apply Spectral Normalization
        model.apply(add_sn)
        model.to(device)
        state_dict = os.path.join('checkpoints/celeba'+str(i), pretrained_weights[i])
        model.load_state_dict(torch.load(state_dict, map_location=device), strict=False)
        model.eval()


        print('-------client{}-------'.format(str(i)))
        image_path = os.path.join(opt.image_path, str(i))
        save_path = os.path.join(opt.save_path, str(i))

        print('image path: ' + image_path)
        print('state dict: ' + state_dict)
        print('attribute vectors: ' + attribute_vectors)
        print('save path: ' + save_path)

        male_attribute_vectors_file = os.path.join(attribute_vectors, 'male_attribute_vectors_'+str(i)+'.t')
        female_attribute_vectors_file = os.path.join(attribute_vectors,             'female_attribute_vectors_' + str(i) + '.t')

        male_attribute_vectors, female_attribute_vectors = compute_attribute_vector(model, image_path, opt.img_sz, attrs, data_attrs, male_attribute_vectors_file, female_attribute_vectors_file, device)
        trans_attributes(model, image_path, save_path, male_attribute_vectors, female_attribute_vectors, male_attribute_vectors_file, female_attribute_vectors_file, attrs, device)


if __name__ == '__main__':
    main()
