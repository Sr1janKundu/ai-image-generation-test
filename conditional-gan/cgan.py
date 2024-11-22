import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable



def train(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=int, default=2e-4, help="learning rate for adam optimizer")
    parser.add_argument("--beta1", type=float, default=0.5, help="decay of first order momentum of gradient for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="decay of second order momentum of gradient for Adam optimizer")
    parser.add_argument("--latent-dim", type=int, default=100, help="latent space dimensionality")
    parser.add_argument("--num-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--img-size", type=int, default=32, help="image height/width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample-interval", type=int, default=400, help="interval between image sampling")

    opt = parser.parse_args()
    print(f"\nSelected options: {opt}")
    print('\n--------------')

    train(opt)
