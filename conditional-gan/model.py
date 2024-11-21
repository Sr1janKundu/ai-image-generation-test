"""
Conditional GAN paper implementation (https://arxiv.org/abs/1411.1784)
[Model based on w-gan-gp]
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        """

        Args:
            channels_img (): For images
            features_d (int): Channels that will change through the discriminator
            num_classes ():
            img_size ():
        """
        super(Critic, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # Input shape: (N x channels_img x 64 x 64)
            nn.Conv2d(
                channels_img+1,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),              # no batchnorm on first block of discriminator
            self._block(features_d,
                        features_d * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1),     # 16x16
            self._block(features_d * 2,
                        features_d * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1),     # 8x8
            self._block(features_d * 4,
                        features_d * 8,
                        kernel_size=4,
                        stride=2,
                        padding=1),     # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),      # 1x1
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """

        Args:
            self ():
            in_channels ():
            out_channels ():
            kernel_size ():
            stride ():
            padding ():

        Returns:

        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)    # N x C x img_size x img_size
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):        # features_g = 64 makes features_g*16 = 1024
        """

        Args:
            z_dim ():
            channels_img ():
            features_g ():
            num_classes ():
            img_size ():
            embed_size ():
        """
        super(Generator, self).__init__()
        self.img_siz = img_size
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim+embed_size,
                        features_g * 16,
                        kernel_size=4,
                        stride=1,
                        padding=0),     # N x f_g*16 x 4 x 4
            self._block(features_g * 16,
                        features_g * 8,
                        kernel_size=4,
                        stride=2,
                        padding=1),     # 8x8
            self._block(features_g * 8,
                        features_g * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1),      # 16x16
            self._block(features_g * 4,
                        features_g * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1),     # 32x32
            nn.ConvTranspose2d(features_g * 2,
                               channels_img,
                               kernel_size=4,
                               stride=2,
                               padding=1),      # no batchnorm on last block of generator
            nn.Tanh(),      # [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """

        Args:
            in_channels ():
            out_channels ():
            kernel_size ():
            stride ():
            padding ():

        Returns:

        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,     # while using batchnorm, no need to use bias
            ),
            nn.InstanceNorm2d(out_channels, affine=True),       # affine=True ensures learnable parameters
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def init_weights(model):
    """
    Model weight initialization according to paper
    Args:
        model ():

    Returns:

    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    init_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(z_dim, in_channels, 8)
    init_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")

if __name__ == '__main__':
    test()