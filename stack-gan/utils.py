"""
Some code are taken from original implementation
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel


def reverse_transforms(image_tensor):
    """
    Reverse the image transformations for viewing and logging
    Args:
        image_tensor ():

    Returns:

    """
    # print("\nGenerated image stats:", torch.min(image_tensor), torch.max(image_tensor))
    # Inverse normalization
    image_tensor = image_tensor * 0.5 + 0.5  # Reverse normalization

    # Convert to uint8
    image_tensor = (image_tensor * 255).clamp(0, 255).byte()

    # Convert back to a numpy array for display
    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
    # print("\nTransformed image stats:", np.min(image_array), np.max(image_array))
    return image_array


def get_sentence_embeddings(sentences,
                            tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                            model=DistilBertModel.from_pretrained("distilbert-base-uncased"),
                            padding=True,
                            truncation=True,
                            max_length=50):
    """
    Generate sentence embeddings

    Args:
        max_length ():
        truncation ():
        padding ():
        model (): Defaults to DistilBertModel base uncased
        tokenizer (): Defaults to DistilBertTokenizer base uncased
        sentences (list of str): List of sentences to encode.

    Returns:
        torch.Tensor: Sentence embeddings as a 2D tensor of shape (len(sentences), hidden_size).
    """

    # Tokenize and encode sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=padding, truncation=truncation, max_length=max_length)
    model = model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the CLS token embeddings as sentence embeddings
    # For DistilBert,the [CLS] token is a special token that is prepended to the input sequence during tokenization.
    # It is designed to act as a sequence-level embedding, representing the entire input sentence or document. e.g.
    # Input: "A bird with yellow feathers."
    # Tokenized: [CLS] A bird with yellow feathers [SEP]
    # [CLS]: Special token representing the whole sequence.
    # [SEP]: Special token marking the end of the sequence (useful in sentence-pair tasks).
    embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

    # or, use this which is to capture the vector summarizing the sequence by averaging all token embeddings
    # for the cases when DistilBert is not used and there are no such token as [CLS]
    # embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


def KL_loss(mu, logvar):
    """
    KL Divergence loss (from official implementation)
    Args:
        mu ():
        logvar ():

    Returns:

    """
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(dis, real_imgs, fake_imgs, real_labels, fake_labels, conds, criterion):
    """

    Args:
        dis ():
        real_imgs ():
        fake_imgs ():
        real_labels ():
        fake_labels ():
        conds ():

    Returns:

    """
    conds = conds.detach()
    fake_imgs = fake_imgs.detach()
    real_features = dis(real_imgs)
    fake_features = dis(fake_imgs)

    # real pairs
    # print("\nHi 1")
    real_logits = dis.get_cond_logits(real_features, conds)
    errD_real = criterion(real_logits, real_labels)

    # wrong pairs
    # print("\nHi 2")
    wrong_logits = dis.get_cond_logits(real_features, torch.roll(conds, 1, 0))
    errD_wrong = criterion(wrong_logits, fake_labels)
    # fake pairs

    # print("\nHi 3")
    fake_logits = dis.get_cond_logits(fake_features, conds)
    errD_fake = criterion(fake_logits, fake_labels)

    if dis.get_uncond_logits is not None:
        # print("\nHi 4")
        real_logits = dis.get_uncond_logits(real_features)
        # print("\nHi 5")
        fake_logits = dis.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)

        errD = ((errD_real + uncond_errD_real) / 2. + (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.

    else:
        errD = errD_real + (errD_fake + errD_wrong) / 2.

    return errD #, errD_real[0], errD_wrong[0], errD_fake[0]



def compute_generator_loss(dis, fake_imgs, real_labels, conds, criterion):
    """

    Args:
        dis ():
        fake_imgs ():
        real_labels ():
        conds ():

    Returns:

    """
    conds = conds.detach()
    fake_features = dis(fake_imgs)

    # fake pairs
    fake_logits = dis.get_cond_logits(fake_features, conds)
    errD_fake = criterion(fake_logits, real_labels)

    if dis.get_uncond_logits is not None:
        fake_logits = dis.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake

    return errD_fake


def weights_init(m):
    """
    Weight initialization (from official implementation)
    Args:
        m ():

    Returns:

    Usage:
    net.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def save_checkpoint(
        s1_generator,
        s1_discriminator,
        s2_generator,
        s2_discriminator,
        s1_gen_optimizer,
        s1_disc_optimizer,
        s2_gen_optimizer,
        s2_disc_optimizer,
        learning_rate_1,
        learning_rate_2,
        epoch,
        loss_dict,
        checkpoint_dir='checkpoints',
        filename=None
):
    """
    Save a comprehensive checkpoint for StackGAN training.

    Args:
        s1_generator (torch.nn.Module): The stage 1 generator model
        s1_discriminator (torch.nn.Module): The stage 1 discriminator model
        s2_generator (torch.nn.Module): The stage 2 generator model
        s2_discriminator (torch.nn.Module): The stage 2 discriminator model
        s1_gen_optimizer (torch.optim.Optimizer): Optimizer for the stage 1 generator
        s1_disc_optimizer (torch.optim.Optimizer): Optimizer for the stage 1 discriminator
        s2_gen_optimizer (torch.optim.Optimizer): Optimizer for the stage 2 generator
        s2_disc_optimizer (torch.optim.Optimizer): Optimizer for the stage 2 discriminator
        learning_rate_1 (float): Current learning rate for stage 1
        learning_rate_2 (float): Current learning rate for stage 2
        epoch (int): Current training epoch
        loss_dict (dict): Dictionary of losses to track
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
        filename (str, optional): Custom filename for the checkpoint.
                                  If None, a default naming scheme is used.

    Returns:
        str: Full path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare checkpoint dictionary
    checkpoint = {
        # Model state dictionaries
        's1_generator_state_dict': s1_generator.state_dict(),
        's1_discriminator_state_dict': s1_discriminator.state_dict(),
        's2_generator_state_dict': s2_generator.state_dict(),
        's2_discriminator_state_dict': s2_discriminator.state_dict(),

        # Optimizer state dictionaries
        's1_generator_optimizer_state_dict': s1_gen_optimizer.state_dict(),
        's1_discriminator_optimizer_state_dict': s1_disc_optimizer.state_dict(),
        's2_generator_optimizer_state_dict': s2_gen_optimizer.state_dict(),
        's2_discriminator_optimizer_state_dict': s2_disc_optimizer.state_dict(),

        # Training metadata
        'epoch': epoch,
        'learning_rate': {
            'stage1': {
                'generator': learning_rate_1,
                'discriminator': learning_rate_1
            },
            'stage2': {
                'generator': learning_rate_2,
                'discriminator': learning_rate_2
            },
        },

        # Losses and other tracking metrics
        'losses': loss_dict,

        # Additional training context (optional)
        'training_config': {
            'batch_size': None,
            'dataset': None,
            'model_config': None
        }
    }

    # Generate filename if not provided
    if filename is None:
        filename = f'stackgan_checkpoint_epoch_{epoch+1}.pth'

    # Full path for saving
    full_path = os.path.join(checkpoint_dir, filename)

    # Save the checkpoint
    torch.save(checkpoint, full_path)

    print(f"Checkpoint saved to {full_path}")

    return full_path


def load_checkpoint(
        checkpoint_path,
        s1_generator,
        s1_discriminator,
        s2_generator,
        s2_discriminator,
        s1_gen_optimizer,
        s1_disc_optimizer,
        s2_gen_optimizer,
        s2_disc_optimizer,
        device=None
):
    """
    Load a comprehensive checkpoint for StackGAN training.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        s1_generator (torch.nn.Module): The stage 1 generator model to load state into
        s1_discriminator (torch.nn.Module): The stage 1 discriminator model to load state into
        s2_generator (torch.nn.Module): The stage 2 generator model to load state into
        s2_discriminator (torch.nn.Module): The stage 2 discriminator model to load state into
        s1_gen_optimizer (torch.optim.Optimizer): Optimizer for the stage 1 generator
        s1_disc_optimizer (torch.optim.Optimizer): Optimizer for the stage 1 discriminator
        s2_gen_optimizer (torch.optim.Optimizer): Optimizer for the stage 2 generator
        s2_disc_optimizer (torch.optim.Optimizer): Optimizer for the stage 2 discriminator
        device (torch.device, optional): Device to load the checkpoint on

    Returns:
        dict: Loaded checkpoint information with additional metadata
    """
    # Load the checkpoint (with optional device specification)
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)

    # Load model state dictionaries
    s1_generator.load_state_dict(checkpoint['s1_generator_state_dict'])
    s2_generator.load_state_dict(checkpoint['s2_generator_state_dict'])
    s1_discriminator.load_state_dict(checkpoint['s1_discriminator_state_dict'])
    s2_discriminator.load_state_dict(checkpoint['s2_discriminator_state_dict'])

    # Load optimizer state dictionaries
    s1_gen_optimizer.load_state_dict(checkpoint['s1_generator_optimizer_state_dict'])
    s2_gen_optimizer.load_state_dict(checkpoint['s2_generator_optimizer_state_dict'])
    s1_disc_optimizer.load_state_dict(checkpoint['s1_discriminator_optimizer_state_dict'])
    s2_disc_optimizer.load_state_dict(checkpoint['s2_discriminator_optimizer_state_dict'])

    # Restore learning rates (if available)
    learning_rates = checkpoint.get('learning_rate', {})

    # Additional restoration steps
    restored_info = {
        'epoch': checkpoint.get('epoch', 0),
        'learning_rates': {
            's1_generator': learning_rates['stage1'].get('generator', 2e-4),
            's2_generator': learning_rates['stage2'].get('generator', 2e-4),
            's1_discriminator': learning_rates['stage1'].get('discriminator', 2e-4),
            's2_discriminator': learning_rates['stage2'].get('discriminator', 2e-4),
        },
        'losses': checkpoint.get('losses', {}),
        'training_config': checkpoint.get('training_config', {})
    }

    # Optional: Update optimizer learning rates if needed
    for param_group in s1_gen_optimizer.param_groups:
        if restored_info['learning_rates']['s1_generator']:
            # param_group['lr_stage1_gen'] = restored_info['learning_rates']['s1_generator']
            param_group['lr'] = restored_info['learning_rates']['s1_generator']

    for param_group in s2_gen_optimizer.param_groups:
        if restored_info['learning_rates']['s2_generator']:
            # param_group['lr_stage2_gen'] = restored_info['learning_rates']['s2_generator']
            param_group['lr'] = restored_info['learning_rates']['s2_generator']

    for param_group in s1_disc_optimizer.param_groups:
        if restored_info['learning_rates']['s1_discriminator']:
            # param_group['lr_stage2_gen'] = restored_info['learning_rates']['s1_discriminator']
            param_group['lr'] = restored_info['learning_rates']['s1_discriminator']

    for param_group in s2_disc_optimizer.param_groups:
        if restored_info['learning_rates']['s2_discriminator']:
            # param_group['lr_stage2_dis'] = restored_info['learning_rates']['s2_discriminator']
            param_group['lr'] = restored_info['learning_rates']['s2_discriminator']

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resumed from epoch {restored_info['epoch']}")

    return restored_info


def image_grid_for_tb(num_images, images, captions):
    """
    Returns a grid with images and captions for tensorboard plotting
    Args:
        num_images (int): Number of images in batch
        images (torch.tensor): A batch of images (B, C, H, W).
        captions (List[str]): A list of captions for the images.

    Returns:

    """
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))

    if num_images == 1:  # Handle the case for a single image
        axes = [axes]

    for i in range(num_images):
        # Convert image tensor to numpy array for display
        img = reverse_transforms(images[i])

        # Display image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{captions[i]}", fontsize=8)

    plt.tight_layout()
    return fig


def tb_log(stage,
           batch_idx,
           loss_gen,
           loss_dis,
           real_img,
           caps,
           fake_img,
           tb_step,
           writer):
    """
    Helper function to log metrics and images for a specific stage

    Args:
        stage (str): 'stage1' or 'stage2'
        batch_idx (int): Current batch index
        loss_gen (float): Generator loss
        loss_dis (float): Discriminator loss
        real_img (torch.Tensor): Real images
        caps (): Captions
        fake_img (torch.Tensor): Generated fake images
        tb_step (int): Global TensorBoard step counter
        writer (SummaryWriter): TensorBoard writer
    """
    # Log losses every iteration
    writer.add_scalar(f'{stage}/generator_loss', loss_gen, tb_step)
    writer.add_scalar(f'{stage}/discriminator_loss', loss_dis, tb_step)

    # Log images and captions every 50 iterations
    # if batch_idx % 5 == 0:
    #     # Take first 5 images, captions, and generated images
    num_images = min(5, real_img.size(0))
    with torch.no_grad():
        fig_real = image_grid_for_tb(num_images, real_img[:num_images], caps[:num_images])
        fig_fake = image_grid_for_tb(num_images, fake_img[:num_images], caps[:num_images])
        writer.add_figure("Real", fig_real, global_step=tb_step, close=True)
        writer.add_figure("Fake", fig_fake, global_step=tb_step, close=True)