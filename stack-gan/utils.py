"""
Some code are taken from original implementation
"""

import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel


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
        learning_rate,
        epoch,
        loss_dict,
        s1_batch_size,
        s2_batch_size,
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
        learning_rate (float): Current learning rate
        epoch (int): Current training epoch
        loss_dict (dict): Dictionary of losses to track
        s1_batch_size (int): Batch size for stage 1 gan
        s2_batch_size (int): Batch size for stage 2 gan
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
            'generator': learning_rate,
            'discriminator': learning_rate  # Assuming same LR for both, modify if different
        },

        # Losses and other tracking metrics
        'losses': loss_dict,

        # Additional training context (optional)
        'training_config': {
            's1_batch_size': s1_batch_size,
            's2_batch_size': s2_batch_size,
            'dataset': None,
            'model_config': None
        }
    }

    # Generate filename if not provided
    if filename is None:
        filename = f'stackgan_checkpoint_epoch_{epoch}.pth'

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
        generator (torch.nn.Module): The generator model to load state into
        discriminator (torch.nn.Module): The discriminator model to load state into
        generator (torch.nn.Module): The generator model to load state into
        discriminator (torch.nn.Module): The discriminator model to load state into
        gen_optimizer (torch.optim.Optimizer): Optimizer for the generator
        disc_optimizer (torch.optim.Optimizer): Optimizer for the discriminator
        gen_optimizer (torch.optim.Optimizer): Optimizer for the generator
        disc_optimizer (torch.optim.Optimizer): Optimizer for the discriminator
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
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Load optimizer state dictionaries
    gen_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    disc_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    # Restore learning rates (if available)
    learning_rates = checkpoint.get('learning_rate', {})

    # Additional restoration steps
    restored_info = {
        'epoch': checkpoint.get('epoch', 0),
        'learning_rates': {
            'generator': learning_rates.get('generator', None),
            'discriminator': learning_rates.get('discriminator', None)
        },
        'losses': checkpoint.get('losses', {}),
        'training_config': checkpoint.get('training_config', {})
    }

    # Optional: Update optimizer learning rates if needed
    # Note: This depends on your specific optimizer setup
    # For example, with SGD or Adam:
    for param_group in gen_optimizer.param_groups:
        if restored_info['learning_rates']['generator']:
            param_group['lr'] = restored_info['learning_rates']['generator']

    for param_group in disc_optimizer.param_groups:
        if restored_info['learning_rates']['discriminator']:
            param_group['lr'] = restored_info['learning_rates']['discriminator']

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resumed from epoch {restored_info['epoch']}")

    return restored_info

def tb_log(*args):
    pass