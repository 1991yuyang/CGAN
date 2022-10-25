import torch as t
from torch import nn
from torch.nn import functional as F

act = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class FC(nn.Module):

    def __init__(self, in_features, out_features, is_bn, act_name):
        super(FC, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=not is_bn)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm1d(num_features=out_features))
        if act_name.lower() == "leaky":
            self.block.add_module("act", nn.LeakyReLU(0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self, noise_dim, class_num, g_feature_count, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.process_condittion = nn.Sequential(
            FC(in_features=class_num, out_features=32, is_bn=True, act_name="leaky"),
            FC(in_features=32, out_features=64, is_bn=True, act_name="leaky"),
            FC(in_features=64, out_features=class_num, is_bn=True, act_name="leaky")
        )
        self.model = nn.Sequential()
        self.class_num = class_num
        in_features = noise_dim + class_num
        for i, feature_count in enumerate(g_feature_count):
            out_features = feature_count
            self.model.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="leaky"))
            in_features = out_features
        self.model.add_module("fc_final", FC(in_features=out_features, out_features=3 * img_size ** 2, is_bn=False, act_name="tanh"))

    def forward(self, x, condition):
        one_hot_result = F.one_hot(condition, num_classes=self.class_num).type(t.FloatTensor).to(x.device)
        concate_result = t.cat([x, self.process_condittion(one_hot_result)], dim=1).type(t.FloatTensor).to(x.device)
        ret = self.model(concate_result)
        return ret.view((ret.size()[0], 3, self.img_size, self.img_size))


class Discriminator(nn.Module):

    def __init__(self, d_feature_count, img_size, class_num):
        super(Discriminator, self).__init__()
        self.class_num = class_num
        self.img_size = img_size
        self.model = nn.Sequential()
        in_features = img_size ** 2 * 3 + class_num
        self.process_condittion = nn.Sequential(
            FC(in_features=class_num, out_features=32, is_bn=True, act_name="relu"),
            FC(in_features=32, out_features=64, is_bn=True, act_name="relu"),
            FC(in_features=64, out_features=class_num, is_bn=True, act_name="relu")
        )
        for i, feature_count in enumerate(d_feature_count):
            out_features = feature_count
            self.model.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="relu"))
            in_features = out_features
        self.model.add_module("fc_final", FC(in_features=out_features, out_features=1, is_bn=False, act_name="sigmoid"))

    def forward(self, x, condition):
        one_hot_result = F.one_hot(condition, num_classes=self.class_num).type(t.FloatTensor).to(x.device)
        x = x.view((x.size()[0], -1))
        concate_result = t.cat([x, self.process_condittion(one_hot_result)], dim=1).type(t.FloatTensor).to(x.device)
        ret = self.model(concate_result)
        return ret


if __name__ == "__main__":
    from numpy import random as rd
    from loss import GLoss, DLoss
    gcriterion = GLoss(noise_dim=100)
    dcriterion = DLoss(noise_dim=100, class_num=5)
    g_input = t.randn(2, 100)
    condition = t.from_numpy(rd.randint(0, 5, 2)).type(t.LongTensor)
    g = Generator(noise_dim=100, class_num=5, g_feature_count=[32, 64, 128], img_size=256)
    d = Discriminator(d_feature_count=[128, 64, 32], img_size=256, class_num=5)
    g_output = g(g_input, condition)
    d_output = d(g_output, condition)
    real_img = t.randn(2, 3, 256, 256)
    gloss = gcriterion(d, g, condition)
    dloss = dcriterion(d, g, real_img, condition)
    print(dloss)