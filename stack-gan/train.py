"""
Training of StackGAN
"""
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import model, config, utils


def train(
        epoch,
        gen_s1,
        dis_s1,
        gen_s2,
        dis_s2,
        opt_gen_s1,
        opt_dis_s1,
        opt_gen_s2,
        opt_dis_s2,
        g_s1_scaler,
        d_s1_scaler,
        g_s2_scaler,
        d_s2_scaler,
        criterion,
        lr_scheduler_gen_s1,
        lr_scheduler_dis_s1,
        lr_scheduler_gen_s2,
        lr_scheduler_dis_s2,
        loader_s1,
        loader_s2,
        train_stage_2,
        tb_step,
        writer,
):
    """

    Args:
        epoch (): Total number of epoches
        gen_s1 (): Stage 1 generator class
        dis_s1 (): Stage 1 discriminator class
        gen_s2 (): Stage 2 generator class
        dis_s2 (): Stage 2 discriminator class
        opt_gen_s1 (): Optimizer for stage 1 generator
        opt_dis_s1 (): Optimizer for stage 1 discriminator
        opt_gen_s2 (): Optimizer for stage 2 generator
        opt_dis_s2 (): Optimizer for stage 2 discriminator
        g_s1_scaler (): Scaler for stage 1 generator
        d_s1_scaler (): Scaler for stage 1 discriminator
        g_s2_scaler (): Scaler for stage 2 generator
        d_s2_scaler (): Scaler for stage 2 discriminator
        criterion (): Loss function
        lr_scheduler_gen_s1 (): Learning rate scheduler for stage 1 generator
        lr_scheduler_dis_s1 (): Learning rate scheduler for stage 1 discriminator
        lr_scheduler_gen_s2 (): Learning rate scheduler for stage 2 generator
        lr_scheduler_dis_s2 (): Learning rate scheduler for stage 2 discriminator
        loader_s1 (): Dataloader for stage 1
        loader_s2 (): Dataloader for stage 2
        train_stage_2 (bool): Whether to train stage 2 or not
        tb_step (): Tensorboard step
        writer (): Tensorboard writer

    Returns:

    """
    loop = tqdm(loader_s1, leave=True, desc=f"Epoch: {epoch+1}")

    for batch_idx, (img, caps_emb, captions) in enumerate(loop):
        img = img.to(config.device)
        caps_emb = caps_emb.to(config.device)
        curr_batch_size = img.shape[0]

        # fixed noise
        noise_vec = torch.FloatTensor(curr_batch_size, config.hyperparameters['latent_dim']).normal_(0.0, 1.0, generator=torch.manual_seed(42)).to(config.device)

        # train stage 1 discriminator
        with torch.amp.autocast('cuda'):
            _, img_stage1, mu_1, sig_1 = gen_s1(caps_emb, noise_vec)
            # for real images
            dis_s1_real = dis_s1(img, caps_emb)

            # for fake images
            dis_s1_fake_g1 = dis_s1(img, torch.roll(caps_emb, 1, 0))
            dis_s1_fake_g2 = dis_s1(img_stage1.detach(), caps_emb)      # .detach() in order to reuse dis_s1_real in calculating stage1 generator loss

            # positive labels
            pos_labels = torch.ones_like(dis_s1_real).to(config.device)
            # negative labels
            neg_labels = torch.zeros_like(dis_s1_fake_g1).to(config.device)

            # calculate stage 1 discriminator loss
            # positive sample pairs
            stage1dis_pos_loss = criterion(dis_s1_real, pos_labels)

            # negative sample pairs
            # group 1: real images with mismatched text embeddings
            stage1dis_neg_loss_grp1 = criterion(dis_s1_fake_g1, neg_labels)
            # group 2: synthetic images with their corresponding text embeddings
            stage1dis_neg_loss_grp2 = criterion(dis_s1_fake_g2, neg_labels)

            # stage 1 discriminator loss
            loss_dis_s1 = stage1dis_pos_loss + (stage1dis_neg_loss_grp1 + stage1dis_neg_loss_grp2) * 0.5

        opt_dis_s1.zero_grad()
        d_s1_scaler.scale(loss_dis_s1).backward()
        d_s1_scaler.step(opt_dis_s1)
        d_s1_scaler.update()

        # train stage 1 generator
        with torch.amp.autocast('cuda'):
            dis_s1_fake = dis_s1(img_stage1, caps_emb)
            # calculate loss
            stage1_neg_loss = criterion(dis_s1_fake, pos_labels)
            stage1_kl_div = utils.KL_loss(mu=mu_1, logvar=sig_1)

            # stage 1 generator loss
            loss_gen_s1 = stage1_neg_loss + config.hyperparameters['gen_loss_kld_reg_param'] * stage1_kl_div

        opt_gen_s1.zero_grad()
        g_s1_scaler.scale(loss_gen_s1).backward()
        g_s1_scaler.step(opt_gen_s1)
        g_s1_scaler.update()

    # update learning rate
    lr_scheduler_gen_s1.step()
    lr_scheduler_dis_s1.step()


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StackGAN training")
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()

    main(parser.parse_args())

"""
To-Do:
-----

1. The training function:
    1.1. Implement tensorboard logging: Plot generator loss, discriminator loss, plot real images and captions, plot fake images and captions 
    (should also take start epoch and end epoch as input)
    1.2. Add code for stage 2 generator and discriminator training
    
2. The main function requirements:
    2.1. Required args:
        1. load_model (bool)
        2. start epoch (int)
        3. end epoch (int)
        4. 
"""