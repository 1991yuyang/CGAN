from models import Generator, Discriminator
from data_loader import create_data_loader
import torch as t
from torch import nn, optim
import os
from loss import GLoss, DLoss
from numpy import random as rd
import numpy as np
import shutil
from PIL import Image


def clip_weight(model, min, max):
    for param in model.parameters():
        t.clamp_(param.data, min, max)
    return model


def train(g, d, g_optimizer, d_optimizer, loader, d_criterion, g_criterion, current_epoch, class_2_idx):
    global total_step
    idx_2_class = {v:k for k, v in class_2_idx.items()}
    steps = len(loader)
    current_step = 1
    for real_img, real_condition in loader:
        # train discriminator
        real_img_cuda = real_img.cuda(device_ids[0])
        real_condition_cuda = real_condition.cuda(device_ids[0])
        d_loss = d_criterion(d, g, real_img_cuda, real_condition_cuda)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        if use_clip_weight:
            d = clip_weight(d, clip_weight_min, clip_weight_max)
        if total_step % d_train_times == 0:
            # train generator
            fake_condition_cuda = t.from_numpy(rd.randint(0, class_num, batch_size)).type(t.LongTensor).cuda(device_ids[0])
            g_loss = g_criterion(d, g, fake_condition_cuda)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            d_optimizer.zero_grad()
            print("epoch:%d/%d, step:%d/%d, g_loss:%.5f, d_loss:%.5f" % (current_epoch, epoch, current_step, steps, g_loss.item(), d_loss.item()))
        if total_step % test_step == 0:
            img_save_pth = os.path.join(test_save_img_dir, str(total_step))
            noise = t.from_numpy(rd.normal(0, 1, (batch_size, noise_dim))).type(t.FloatTensor).cuda(device_ids[0])
            fake_condition_cuda = t.from_numpy(rd.randint(0, class_num, batch_size)).type(t.LongTensor).cuda(device_ids[0])
            fake_imgs = (np.transpose(g(noise, fake_condition_cuda).cpu().detach().numpy(), axes=[0, 2, 3, 1]) * 127.5 + 127.5).astype(np.uint8)
            fake_condition_list = fake_condition_cuda.cpu().detach().numpy().tolist()
            os.mkdir(os.path.join(test_save_img_dir, str(total_step)))
            for i in range(batch_size):
                if is_gray:
                    fake_img = fake_imgs[i, :, :, 0]
                else:
                    fake_img = fake_imgs[i]
                condition = fake_condition_list[i]
                class_name = idx_2_class[condition]
                pil_save_pth = os.path.join(img_save_pth, "%s.jpg" % class_name)
                pil_img = Image.fromarray(fake_img)
                pil_img.save(pil_save_pth)
        total_step += 1
        current_step += 1
    return d, g


def main():
    g = Generator(noise_dim, class_num, g_feature_count, img_size, is_gray)
    g = nn.DataParallel(module=g, device_ids=device_ids)
    g = g.cuda(device_ids[0])
    d = Discriminator(d_feature_count, img_size, class_num, is_gray)
    d = nn.DataParallel(module=d, device_ids=device_ids)
    d = d.cuda(device_ids[0])
    d_criterion = DLoss(noise_dim, class_num).cuda(device_ids[0])
    g_criterion = GLoss(noise_dim).cuda(device_ids[0])
    d_optimizer = optim.Adam(params=d.parameters(), lr=d_init_lr)
    g_optimizer = optim.Adam(params=g.parameters(), lr=g_init_lr)
    d_lr_sch = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epoch, eta_min=d_final_lr)
    g_lr_sch = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epoch, eta_min=d_final_lr)
    for e in range(epoch):
        current_epoch = e + 1
        loader, class_2_idx = create_data_loader(data_root_dir, img_size, num_workers, batch_size, is_gray)
        d, g = train(g, d, g_optimizer, d_optimizer, loader, d_criterion, g_criterion, current_epoch, class_2_idx)
        d_lr_sch.step()
        g_lr_sch.step()


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    data_root_dir = r"F:\data\mnist"
    epoch = 500
    batch_size = 64
    noise_dim = 128
    d_init_lr = 1e-5
    d_final_lr = 1e-6
    g_init_lr = 1e-4
    g_final_lr = 1e-5
    img_size = 28
    g_feature_count = [128, 256, 512]
    d_train_times = 1
    test_step = 1000
    num_workers = 4
    use_clip_weight = True
    clip_weight_min = -0.15
    clip_weight_max = 0.15
    is_gray = True
    test_save_img_dir = r"test_images"
    if os.path.exists(test_save_img_dir):
        shutil.rmtree(test_save_img_dir)
    os.mkdir(test_save_img_dir)
    d_feature_count = g_feature_count[::-1]
    total_step = 1
    device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
    class_num = len(os.listdir(data_root_dir))
    main()
