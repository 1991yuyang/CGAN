import torch as t
from torch import nn


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
        self.block.add_module("act", act[act_name]())

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    def __init__(self, noise_dim, class_num, g_feature_count, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential()
        in_features = noise_dim + class_num
        for i, feature_count in enumerate(g_feature_count):
            out_features = feature_count
            self.model.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="relu"))
            in_features = out_features
        self.model.add_module("fc_final", FC(in_features=out_features, out_features=3 * img_size ** 2, is_bn=False, act_name="tanh"))

    def forward(self, x):
        ret = self.model(x)
        return ret.view((ret.size()[0], 3, self.img_size, self.img_size))


class Discriminator(nn.Module):

    def __init__(self, d_feature_count, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential()
        in_features = img_size ** 2 * 3
        for i, feature_count in enumerate(d_feature_count):
            out_features = feature_count
            self.model.add_module("fc_%d" % (i,), FC(in_features=in_features, out_features=out_features, is_bn=True, act_name="relu"))
            in_features = out_features
        self.model.add_module("fc_final", FC(in_features=out_features, out_features=1, is_bn=False, act_name="sigmoid"))

    def forward(self, x):
        x = x.view((x.size()[0], -1))
        ret = self.model(x)
        return ret


if __name__ == "__main__":
    g_input = t.randn(2, 105)
    g = Generator(noise_dim=100, class_num=5, g_feature_count=[32, 64, 128], img_size=256)
    d = Discriminator(d_feature_count=[128, 64, 32], img_size=256)
    g_output = g(g_input)
    d_output = d(g_output)
    print(g_output.size())
    print(d_output.size())