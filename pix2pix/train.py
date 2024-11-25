"""
Implementation of pix2pix paper (Image-to-Image Translation with Conditional Adversarial Networks)
(https://arxiv.org/abs/1611.07004)

"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from gitdb.util import exists
from tqdm import tqdm

from utils import save_checkpoint, load_checkpoint, save_some_examples
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader


def train_fn(epoch, disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    """

    Args:
        epoch ():
        disc ():
        gen ():
        loader ():
        opt_disc ():
        opt_gen ():
        l1 ():
        bce ():
        g_scaler ():
        d_scaler ():

    Returns:

    """
    loop = tqdm(loader, leave=True, desc= f"Epoch {epoch + 1}")

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train discriminator
        with torch.amp.autocast('cuda'):         # for float16 training
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            real_labels = torch.ones_like(D_real).to(config.DEVICE)
            D_real_loss = bce(D_real, real_labels)
            fake_labels = torch.zeros_like(D_fake).to(config.DEVICE)
            D_fake_loss = bce(D_fake, fake_labels)
            D_loss = (D_real_loss + D_fake_loss) / 2        # try without /2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()       # do retain_graph=True, if not done y_fake.detach() previously while calculating D_fake
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.amp.autocast('cuda'):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake).to(config.DEVICE))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))       # try with beta1 = 0.9
    opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()    # standard GAN loss
    L1_loss = nn.L1Loss()       # wgangp loss did not work well with patchgan

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir="../dataset/pix2pix_datasets/maps/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # to train on float16
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    val_dataset = MapDataset(root_dir="../dataset/pix2pix_datasets/maps/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(epoch, disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
