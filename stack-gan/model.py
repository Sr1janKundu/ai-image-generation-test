"""
Implementation of StackGan model.
paper: https://arxiv.org/abs/1612.03242
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
# import torch.nn.functional as F
# from torch.autograd import Variable

import config


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


class ConditionalAugmentation(nn.Module):
    def __init__(self, embedding_dim:int=768, output_dim:int=128):
        """
        Conditional augmentation class to generate embedding vectors for captions
        Args:
            embedding_dim (int): Pre-trained embedding vector dimension; defaults to 768 which is embedding size of DistilBert
            output_dim (int):
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # self.fc_mu = nn.Linear(embedding_dim, output_dim)
        # self.fc_sigma = nn.Linear(embedding_dim, output_dim)
        self.fc = nn.Linear(embedding_dim, output_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        """
        To get mu and sigma from text embedding
        Args:
            text_embedding ():

        Returns:

        """
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.output_dim]
        logvar = x[:, self.output_dim:]

        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Get dimensional conditioning vector from mu and logvar
        Args:
            mu ():
            logvar ():

        Returns:

        """
        std = logvar.mul(0.5).exp_()                        # .exp_() does in-place exponential of the elements
        eps = torch.cuda.FloatTensor(std.size()).normal_() if config.device == "cuda" else torch.FloatTensor(std.size()).normal_()
        # eps = eps.requires_grad_()
        # eps = Variable(eps)

        # print(eps.requires_grad)

        return eps.mul(std).add_(mu)

    def forward(self, embedding_vec):
        """
        One can directly generate variance with the following commented-out part, but it would be more standard approach
        to model uncertainty with the logvar approach from the VAE paper implementation (https://github.com/pytorch/examples/blob/master/vae/main.py)
        Args:
            embedding_vec ():

        Returns:

        """
        # mu0 = self.relu(self.fc_mu(embedding_vec))
        # sigma0 = self.fc_sigma(embedding_vec)
        #
        # # Apply softplus to ensure positive standard deviation
        # sigma0 = F.softplus(sigma0)
        #
        # # Re-parameterization
        # eps0 = torch.rand_like(sigma0)
        #
        # conditioning_vec = mu0 + eps0 * sigma0

        mu0, logvar0 = self.encode(embedding_vec)
        conditioning_vec = self.reparameterize(mu0, logvar0)

        return conditioning_vec, mu0, logvar0


def conv3x3(in_channels, out_channels, stride=1):
    """

    Args:
        in_channels ():
        out_channels ():
        stride ():

    Returns:

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def UpBlock(in_channels, out_channels):
    """

    Args:
        in_channels ():
        out_channels ():

    Returns:

    """
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=config.hyperparameters['upsampling_mode']),
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return block


class GeneratorStage1(nn.Module):
    """
    The stage 1 Generator
    """
    def __init__(self, conditional_vec_dim, latent_dim, ):
        super().__init__()
        pass



class GeneratorStage2(nn.Module):
    """
    The stage 2 Generator
    """
    pass


class DiscriminatorStage1(nn.Module):
    """
    The stage 1 Discriminator
    """
    pass


class DiscriminatorStage2(nn.Module):
    """
    The stage 2 Discriminator
    """
    pass


def test():
    embedding_dim = 768  # Dimensionality of text embedding (DistilBERT output size)
    conditioning_dim = 128  # Dimensionality of the conditioning vector (Ng)
    text_embedding = torch.randn(16, embedding_dim)

    # Initialize the conditioning network
    text_conditioning = ConditionalAugmentation(embedding_dim, conditioning_dim)

    # Generate the conditioning vector using the reparameterization trick
    conditioning_vector, _, _ = text_conditioning(text_embedding)

    print(f"Conditioning Vector Shape: {conditioning_vector.shape}")
    print("---Conditional augmentation step passed---")


if __name__=="__main__":
    test()