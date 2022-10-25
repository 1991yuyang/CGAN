import torch as t
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as T
import numpy as np


"""
data_root_dir
    class1
        img1.jpg
        img2.jpg
        ...
    class2
        img1.jpg
        img2.jpg
        ...
    ...
"""


class ToTensorTan(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        img = (img - 127.5) / 127.5
        img = t.tensor(img).permute(dims=[2, 0, 1])
        return img


def create_data_loader(data_root_dir, img_size, num_workers, batch_size):
    ds = datasets.ImageFolder(data_root_dir, transform=T.Compose([
        T.Resize((img_size, img_size)),
        ToTensorTan()
    ]))
    class_to_idx = ds.class_to_idx
    data_loader = data.DataLoader(ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader, class_to_idx


if __name__ == "__main__":
    data_root_dir = r"F:\data\animal\animal image dataset\animals\animals"
    loader, class_to_idx = create_data_loader(data_root_dir, 256, 4, 8)
    print(class_to_idx)
    for d, l in loader:
        print(d.size())
        print(l)