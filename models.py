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


class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn, act_name, kernel_size):
        super(DeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 2) // 2, bias=not is_bn)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        if act_name.lower() == "leaky":
            self.block.add_module("act", nn.LeakyReLU(0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn, act_name, kernel_size):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2 - 1, bias=not is_bn)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        if act_name.lower() == "leaky":
            self.block.add_module("act", nn.LeakyReLU(0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        return self.block(x)


class PWConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn, act_name):
        super(PWConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=not is_bn),
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        if act_name.lower() == "leaky":
            self.block.add_module("act", nn.LeakyReLU(0.2))
        else:
            self.block.add_module("act", act[act_name]())

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self, noise_dim, class_num, g_feature_count, img_size, is_gray):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.img_size = img_size
        self.is_gray = is_gray
        self.class_num = class_num
        self.process_condition = nn.Sequential(
            FC(in_features=class_num, out_features=32, is_bn=True, act_name="leaky"),
            FC(in_features=32, out_features=64, is_bn=True, act_name="leaky"),
            FC(in_features=64, out_features=noise_dim, is_bn=True, act_name="leaky")
        )
        self.model = nn.Sequential()
        in_features = noise_dim * 2
        for i, feature_count in enumerate(g_feature_count):
            out_features = feature_count
            self.model.add_module("deconv_%d" % (i,), DeConvBlock(in_channels=in_features, out_channels=out_features, is_bn=True, act_name="leaky", kernel_size=4))
            in_features = out_features
        if is_gray:
            self.model.add_module("conv_final", PWConv(in_channels=out_features, out_channels=1, is_bn=False, act_name="tanh"))
        else:
            self.model.add_module("conv_final", PWConv(in_channels=out_features, out_channels=3, is_bn=False, act_name="tanh"))

    def forward(self, x, condition):
        one_hot_result = F.one_hot(condition, num_classes=self.class_num).type(t.FloatTensor).to(x.device)
        condition_feature = self.process_condition(one_hot_result).view((x.size()[0], self.noise_dim, 1, 1))
        concate_result = t.cat([x, condition_feature], dim=1).type(t.FloatTensor).to(x.device)
        ret = self.model(concate_result)
        return ret


class Discriminator(nn.Module):

    def __init__(self, d_feature_count, img_size, class_num, is_gray):
        super(Discriminator, self).__init__()
        self.class_num = class_num
        self.img_size = img_size
        self.model = nn.Sequential()
        if is_gray:
            in_features = 1 * 2
        else:
            in_features = 3 * 2
        self.process_condition = nn.Sequential(
            FC(in_features=class_num, out_features=32, is_bn=True, act_name="relu"),
            FC(in_features=32, out_features=64, is_bn=True, act_name="relu"),
            FC(in_features=64, out_features=(in_features // 2) * img_size ** 2, is_bn=True, act_name="relu")
        )
        for i, feature_count in enumerate(d_feature_count):
            out_features = feature_count
            self.model.add_module("conv_%d" % (i,), ConvBlock(in_channels=in_features, out_channels=out_features, is_bn=True, act_name="relu", kernel_size=4))
            in_features = out_features
        self.model.add_module("conv_final", PWConv(in_channels=out_features, out_channels=1, is_bn=False, act_name="sigmoid"))

    def forward(self, x, condition):
        one_hot_result = F.one_hot(condition, num_classes=self.class_num).type(t.FloatTensor).to(x.device)
        condition_feature = self.process_condition(one_hot_result).view((x.size()[0], -1, self.img_size, self.img_size))
        concate_result = t.cat([x, condition_feature], dim=1).type(t.FloatTensor).to(x.device)
        ret = self.model(concate_result).view((x.size()[0], 1))
        return ret
