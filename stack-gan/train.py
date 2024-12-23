"""
Training of StackGAN
"""
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import model_dev as model, dataset, config, utils


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

        loop_s2 = tqdm(loader_s2, leave=True, desc=f"Epoch {epoch+1} (stage 2 gan)")

        for batch_idx, (img, cap_emb, captions) in enumerate(loop_s2):

            img = img.to(config.device)
            cap_emb = cap_emb.to(config.device)
            curr_batch_size_s2 = img.shape[0]

            # positive labels
            pos_labels = torch.FloatTensor(curr_batch_size_s2).fill_(1).to(config.device)
            # negative labels
            neg_labels = torch.FloatTensor(curr_batch_size_s2).fill_(0).to(config.device)

            # fixed noise
            noise_vec = torch.FloatTensor(curr_batch_size_s2, config.hyperparameters['latent_dim']).normal_(0.0, 1.0, generator=torch.manual_seed(42)).to(config.device)

            # generate fake images
            _, img_stage2, mu_2, sig_2 = gen_s2(cap_emb, noise_vec)

            # train stage 2 discriminator
            dis_s2.zero_grad()

            # calculate losses for stage 2 discriminator
            errD_stage2 = utils.compute_discriminator_loss(dis_s2, img, img_stage2, pos_labels, neg_labels, mu_2, criterion)
            errD_stage2.backward()
            opt_dis_s2.step()

            # train stage 2 generator
            gen_s2.zero_grad()
            errG_stage2 = utils.compute_generator_loss(dis_s2, img_stage2, pos_labels, mu_2, criterion)
            stage2_kl_div_stage2 = utils.KL_loss(mu=mu_2, logvar=sig_2)
            errG_total_stage2 = errG_stage2 + stage2_kl_div_stage2 * config.hyperparameters['gen_loss_kld_reg_param']
            errG_total_stage2.backward()
            opt_gen_s2.step()

            # logging
            utils.tb_log(stage="stage2",
                         batch_idx=batch_idx,
                         loss_gen=errG_total_stage2.item(),
                         loss_dis=errD_stage2.item(),
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

        loop_s1 = tqdm(loader_s1, leave=True, desc=f"Epoch {epoch+1} (stage 1 gan)")

        for batch_idx, (img, caps_emb, captions) in enumerate(loop_s1):

            img = img.to(config.device)
            caps_emb = caps_emb.to(config.device)
            curr_batch_size_s1 = img.shape[0]

            # positive labels
            pos_labels = torch.FloatTensor(curr_batch_size_s1).fill_(1).to(config.device)
            # negative labels
            neg_labels = torch.FloatTensor(curr_batch_size_s1).fill_(0).to(config.device)

            # fixed noise
            noise_vec = torch.FloatTensor(curr_batch_size_s1, config.hyperparameters['latent_dim']).normal_(0.0, 1.0, generator=torch.manual_seed(42)).to(config.device)

            # generate fake images
            _, img_stage1, mu_1, sig_1 = gen_s1(caps_emb, noise_vec)

            # train stage 1 discriminator
            dis_s1.zero_grad()

            # calculate losses for stage 1 discriminator
            errD_stage1 = utils.compute_discriminator_loss(dis_s1, img, img_stage1, pos_labels, neg_labels, mu_1, criterion)
            errD_stage1.backward()
            opt_dis_s1.step()

            # train stage 1 generator
            gen_s1.zero_grad()
            errG_stage1 = utils.compute_generator_loss(dis_s1, img_stage1, pos_labels, mu_1, criterion)
            stage1_kl_div_stage1 = utils.KL_loss(mu=mu_1, logvar=sig_1)
            errG_total_stage1 = errG_stage1 + stage1_kl_div_stage1 * config.hyperparameters['gen_loss_kld_reg_param']
            errG_total_stage1.backward()
            opt_gen_s1.step()

            # logging
            utils.tb_log(stage="stage1",
                         batch_idx=batch_idx,
                         loss_gen=errG_total_stage1.item(),
                         loss_dis=errD_stage1.item(),
                         real_img=img,
                         caps=captions,
                         fake_img=img_stage1,
                         tb_step=tb_step,
                         writer=writer)
            tb_step += 1

        # update learning rate
        lr_scheduler_gen_s1.step()
        lr_scheduler_dis_s1.step()


    # save everything every 5 epoch
    if (epoch + 1) % 5 == 0:
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
            checkpoint_path=os.path.join('checkpoints', args.checkpoint),
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
    elif config.load_stage2:
        info2 = utils.load_checkpoint(
            checkpoint_path=os.path.join('checkpoints', args.checkpoint),
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
        stage1_dis.apply(utils.weights_init)
        stage2_gen.apply(utils.weights_init)
        stage2_dis.apply(utils.weights_init)

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
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
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
    parser.add_argument('--checkpoint', type=str, help="Name of latest checkpoint file, if config.load_stage1 or config.load_stage2 is True")

    main(parser.parse_args())