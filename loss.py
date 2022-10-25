import torch as t
from torch import nn
from numpy import random as rd


class GLoss(nn.Module):

    def __init__(self, noise_dim):
        super(GLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)
        self.noise_dim = noise_dim

    def forward(self, discriminator, generator, condition):
        generator.train()
        discriminator.eval()
        noise = t.from_numpy(rd.normal(0, 1, (condition.size()[0], self.noise_dim))).type(t.FloatTensor).to(condition.device)
        fake_img = generator(noise, condition)
        fake_d_output = discriminator(fake_img, condition)
        target = t.ones(fake_d_output.size()).type(t.FloatTensor).to(fake_d_output.device)
        loss = self.bce(fake_d_output, target)
        return loss


class DLoss(nn.Module):

    def __init__(self, noise_dim, class_num):
        super(DLoss, self).__init__()
        self.bce = nn.BCELoss().cuda(0)
        self.noise_dim = noise_dim
        self.class_num = class_num

    def forward(self, discriminator, generator, real_img, real_condition):
        discriminator.train()
        generator.eval()
        batch_size = real_img.size()[0]
        noise = t.from_numpy(rd.normal(0, 1, (batch_size, self.noise_dim))).type(t.FloatTensor).to(real_img.device)
        fake_condition = t.from_numpy(rd.randint(0, self.class_num, batch_size)).type(t.LongTensor).cuda(real_condition.device)
        with t.no_grad():
            fake_img = generator(noise, fake_condition)
        fake_target = t.zeros((fake_img.size()[0], 1)).type(t.FloatTensor).to(real_img.device)
        real_target = t.ones((real_img.size()[0], 1)).type(t.FloatTensor).to(real_img.device)
        d_input_img = t.cat([fake_img, real_img], dim=0)
        target = t.cat([fake_target, real_target], dim=0)
        d_output = discriminator(d_input_img, t.cat([fake_condition, real_condition], dim=0))
        loss = self.bce(d_output, target)
        return loss