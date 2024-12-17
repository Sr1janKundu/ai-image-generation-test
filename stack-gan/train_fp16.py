"""
Training of StackGAN
"""
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import model, dataset, config, utils


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
    The training function
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
        tb_step (): Global TensorBoard step counter
        writer (): TensorBoard SummaryWriter instance

    Returns:
        (int): Global tensorboard step after logging
    """

    # train stage 2 gan
    if train_stage_2 and epoch > config.hyperparameters['train_stage1_for']:

        # first freeze stage 1 generator
        for param in gen_s1.parameters():
            param.requires_grad = False

        loop_s2 = tqdm(loader_s2, leave=True, desc=f"Epoch: {epoch+1}")
        loss_dis_s2_avg, loss_gen_s2_avg = 0.0, 0.0
        for batch_idx, (img, cap_emb, captions) in enumerate(loop_s2):
            opt_dis_s2.zero_grad()
            opt_gen_s2.zero_grad()

            img = img.to(config.device)
            cap_emb = cap_emb.to(config.device)
            curr_batch_size_s2 = img.shape[0]

            # fixed noise
            noise_vec = torch.FloatTensor(curr_batch_size_s2, config.hyperparameters['latent_dim']).normal_(0.0, 1.0, generator=torch.manual_seed(42)).to(config.device)

            # train stage 2 discriminator
            with torch.amp.autocast('cuda'):
                _, img_stage2, mu_2, sig_2 = gen_s2(cap_emb, noise_vec)

                # for real images
                dis_s2_real = dis_s2(img, cap_emb)

                # for fake images
                dis_s2_fake_g1 = dis_s2(img, torch.roll(cap_emb, 1, 0))
                dis_s2_fake_g2 = dis_s2(img_stage2.detach(),
                                        cap_emb)  # .detach() in order to reuse dis_s2_real in calculating stage1 generator loss

                # positive labels
                pos_labels = torch.ones_like(dis_s2_real).to(config.device)
                # negative labels
                neg_labels = torch.zeros_like(dis_s2_fake_g1).to(config.device)

                # calculate stage 2 discriminator loss
                # positive sample pairs
                stage2dis_pos_loss = criterion(dis_s2_real, pos_labels)

                # negative sample pairs
                # group 1: real images with mismatched text embeddings
                stage2dis_neg_loss_grp1 = criterion(dis_s2_fake_g1, neg_labels)
                # group 2: synthetic images with their corresponding text embeddings
                stage2dis_neg_loss_grp2 = criterion(dis_s2_fake_g2, neg_labels)

                # stage 2 discriminator loss
                loss_dis_s2 = stage2dis_pos_loss + (stage2dis_neg_loss_grp1 + stage2dis_neg_loss_grp2) * 0.5

            d_s2_scaler.scale(loss_dis_s2).backward()
            d_s2_scaler.step(opt_dis_s2)
            d_s2_scaler.update()

            loss_dis_s2_avg = loss_dis_s2 / len(loader_s2)

            # train stage 2 generator
            with torch.amp.autocast('cuda'):
                dis_s2_fake = dis_s2(img_stage2, cap_emb)
                # calculate loss
                stage2_neg_loss = criterion(dis_s2_fake, pos_labels)
                stage2_kl_div = utils.KL_loss(mu=mu_2, logvar=sig_2)

                # stage 2 generator loss
                loss_gen_s2 = stage2_neg_loss + config.hyperparameters['gen_loss_kld_reg_param'] * stage2_kl_div

            g_s2_scaler.scale(loss_gen_s2).backward()
            g_s2_scaler.step(opt_gen_s2)
            g_s2_scaler.update()

            loss_gen_s2_avg = loss_gen_s2 / len(loader_s2)

            # logging
            utils.tb_log(stage="stage2",
                         batch_idx=batch_idx,
                         loss_gen=loss_gen_s2_avg,
                         loss_dis=loss_dis_s2_avg,
                         real_img=img,
                         caps=captions,
                         fake_img=img_stage2,
                         tb_step=tb_step,
                         writer=writer)
            tb_step += 1

        # update learning rate
        lr_scheduler_gen_s2.step()
        lr_scheduler_dis_s2.step()

    else:   # train stage 1 gan

        loop_s1 = tqdm(loader_s1, leave=True, desc=f"Epoch: {epoch+1}")
        loss_dis_s1_avg, loss_gen_s1_avg = 0.0, 0.0
        for batch_idx, (img, caps_emb, captions) in enumerate(loop_s1):
            opt_dis_s1.zero_grad()
            opt_gen_s1.zero_grad()

            img = img.to(config.device)
            caps_emb = caps_emb.to(config.device)
            curr_batch_size_s1 = img.shape[0]

            # fixed noise
            noise_vec = torch.FloatTensor(curr_batch_size_s1, config.hyperparameters['latent_dim']).normal_(0.0, 1.0, generator=torch.manual_seed(42)).to(config.device)

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

            d_s1_scaler.scale(loss_dis_s1).backward()
            d_s1_scaler.step(opt_dis_s1)
            d_s1_scaler.update()

            loss_dis_s1_avg = loss_dis_s1.item() / len(loader_s1)

            # train stage 1 generator
            with torch.amp.autocast('cuda'):
                dis_s1_fake = dis_s1(img_stage1, caps_emb)
                # calculate loss
                stage1_neg_loss = criterion(dis_s1_fake, pos_labels)
                stage1_kl_div = utils.KL_loss(mu=mu_1, logvar=sig_1)

                # stage 1 generator loss
                loss_gen_s1 = stage1_neg_loss + config.hyperparameters['gen_loss_kld_reg_param'] * stage1_kl_div

            g_s1_scaler.scale(loss_gen_s1).backward()
            g_s1_scaler.step(opt_gen_s1)
            g_s1_scaler.update()

            loss_gen_s1_avg = loss_gen_s1.item() / len(loader_s1)

            # logging
            utils.tb_log(stage="stage1",
                         batch_idx=batch_idx,
                         loss_gen=loss_gen_s1_avg,
                         loss_dis=loss_dis_s1_avg,
                         real_img=img,
                         caps=captions,
                         fake_img=img_stage1,
                         tb_step=tb_step,
                         writer=writer)
            tb_step += 1

        # update learning rate
        lr_scheduler_gen_s1.step()
        lr_scheduler_dis_s1.step()

    # save everything every 10 epoch
    if epoch % 10 == 0:
        save_path = utils.save_checkpoint(
            s1_generator=gen_s1,
            s1_discriminator=dis_s1,
            s2_generator=gen_s2,
            s2_discriminator=dis_s2,
            s1_gen_optimizer=opt_gen_s1,
            s1_disc_optimizer=opt_dis_s1,
            s2_gen_optimizer=opt_gen_s2,
            s2_disc_optimizer=opt_dis_s2,
            learning_rate_1=lr_scheduler_gen_s1.get_last_lr()[0],
            learning_rate_2=lr_scheduler_gen_s2.get_last_lr()[0],
            epoch=epoch,
            loss_dict=dict(),
        )

    return tb_step

def main(args):
    # initialize model
    stage1_gen = model.GeneratorStage1().to(config.device)
    stage1_dis = model.DiscriminatorStage1().to(config.device)
    stage2_gen = model.GeneratorStage2(stage1_gen).to(config.device)
    stage2_dis = model.DiscriminatorStage2().to(config.device)

    # initialize optimizers
    s1_gen_opt = optim.Adam(stage1_gen.parameters(), lr = config.hyperparameters['start_lr'], betas=(0.5, 0.999))
    s1_dis_opt = optim.Adam(stage1_dis.parameters(), lr = config.hyperparameters['start_lr'], betas=(0.5, 0.999))
    s2_gen_opt = optim.Adam(stage2_gen.parameters(), lr = config.hyperparameters['start_lr'], betas=(0.5, 0.999))
    s2_dis_opt = optim.Adam(stage2_dis.parameters(), lr = config.hyperparameters['start_lr'], betas=(0.5, 0.999))

    if config.load_stage1:
        info1 = utils.load_checkpoint(
            checkpoint_path='checkpoints',
            s1_generator=stage1_gen,
            s1_discriminator=stage1_dis,
            s2_generator=stage2_gen,
            s2_discriminator=stage2_dis,
            s1_gen_optimizer=s1_gen_opt,
            s1_disc_optimizer=s1_dis_opt,
            s2_gen_optimizer=s2_gen_opt,
            s2_disc_optimizer=s2_dis_opt,
            device=config.device
        )
    else:
        stage1_gen.apply(utils.weights_init)
        stage1_gen.apply(utils.weights_init)
        stage2_gen = model.GeneratorStage2(stage1_gen).to(config.device)

    if config.load_stage2:
        info2 = utils.load_checkpoint(
            checkpoint_path='checkpoints',
            s1_generator=stage1_gen,
            s1_discriminator=stage1_dis,
            s2_generator=stage2_gen,
            s2_discriminator=stage2_dis,
            s1_gen_optimizer=s1_gen_opt,
            s1_disc_optimizer=s1_dis_opt,
            s2_gen_optimizer=s2_gen_opt,
            s2_disc_optimizer=s2_dis_opt,
            device=config.device
        )
    else:
        stage2_gen.apply(utils.weights_init)
        stage2_gen.apply(utils.weights_init)

    # initialize grad scalars for float16 training
    g_s1_scaler = torch.amp.GradScaler('cuda')
    d_s1_scaler = torch.amp.GradScaler('cuda')
    g_s2_scaler = torch.amp.GradScaler('cuda')
    d_s2_scaler = torch.amp.GradScaler('cuda')

    # loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # initialize lr schedulers
    lr_scheduler_gen_s1 = torch.optim.lr_scheduler.StepLR(
        s1_gen_opt,
        step_size=100,
        gamma=0.5
    )
    lr_scheduler_dis_s1 = torch.optim.lr_scheduler.StepLR(
        s1_dis_opt,
        step_size=100,
        gamma=0.5
    )
    lr_scheduler_gen_s2 = torch.optim.lr_scheduler.StepLR(
        s2_gen_opt,
        step_size=100,
        gamma=0.5
    )
    lr_scheduler_dis_s2 = torch.optim.lr_scheduler.StepLR(
        s2_dis_opt,
        step_size=100,
        gamma=0.5
    )

    # initialize dataloaders
    loader_s1 = dataset.get_flickr8k_loader(
        img_dir=config.image_dir,
        captions_file=config.all_captions_file,
        transform=config.img_trans_stage1,
        text_encoder=utils.get_sentence_embeddings,
        batch_size=config.hyperparameters['batch_size_stage1'],
        shuffle=True,
        num_workers=8
    )

    loader_s2 = dataset.get_flickr8k_loader(
        img_dir=config.image_dir,
        captions_file=config.all_captions_file,
        transform=config.img_trans_stage2,
        text_encoder=utils.get_sentence_embeddings,
        batch_size=config.hyperparameters['batch_size_stage2'],
        shuffle=True,
        num_workers=8
    )

    # tensorboard step
    tb_step = 0

    # tensorboard writer
    writer = SummaryWriter(f"logs/")

    # start training
    start_epoch = (
        info2.get('epoch', 0) if config.load_stage2 else
        info1.get('epoch', 0) if config.load_stage1 else
        0)

    if start_epoch < args.epochs:
        for epoch in range(start_epoch, args.epochs, 1):
            print(f"Epoch [{epoch+1}/{args.epochs}]")
            tb_step = train(
                epoch,
                gen_s1=stage1_gen,
                dis_s1=stage1_dis,
                gen_s2=stage2_gen,
                dis_s2=stage2_dis,
                opt_gen_s1=s1_gen_opt,
                opt_dis_s1=s1_dis_opt,
                opt_gen_s2=s2_gen_opt,
                opt_dis_s2=s2_dis_opt,
                g_s1_scaler=g_s1_scaler,
                d_s1_scaler=d_s1_scaler,
                g_s2_scaler=g_s2_scaler,
                d_s2_scaler=d_s2_scaler,
                criterion=criterion,
                lr_scheduler_gen_s1=lr_scheduler_gen_s1,
                lr_scheduler_dis_s1=lr_scheduler_dis_s1,
                lr_scheduler_gen_s2=lr_scheduler_gen_s2,
                lr_scheduler_dis_s2=lr_scheduler_dis_s2,
                loader_s1=loader_s1,
                loader_s2=loader_s2,
                train_stage_2=args.train_s2,
                tb_step=tb_step,
                writer=writer,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StackGAN training")
    parser.add_argument('--epochs', type=int, default=1200, help="Number of epochs")
    parser.add_argument('--train-s2', type=bool, default=True, help="Whether to train stage 2")

    main(parser.parse_args())



"""
Note:
    Does not work, some problem with torch.amp. Throws the following error:
    Epoch [1/50]
    Epoch: 1:   0%|                                                                                                                                                                                             | 0/253 [00:08<?, ?it/s]
    Traceback (most recent call last):
      File "/home/srijan/Desktop/Srijan-files/ai-image-generation-test/stack-gan/train_fp16.py", line 405, in <module>
        main(parser.parse_args())
      File "/home/srijan/Desktop/Srijan-files/ai-image-generation-test/stack-gan/train_fp16.py", line 373, in main
        tb_step = train(
      File "/home/srijan/Desktop/Srijan-files/ai-image-generation-test/stack-gan/train_fp16.py", line 218, in train
        g_s1_scaler.step(opt_gen_s1)
      File "/home/srijan/anaconda3/envs/dl2/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 453, in step
        assert (
    AssertionError: No inf checks were recorded for this optimizer.

"""