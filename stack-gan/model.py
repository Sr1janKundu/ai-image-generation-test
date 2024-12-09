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
        eps = torch.FloatTensor(std.size()).normal_().to(config.device)
        # eps = eps.requires_grad_()
        # eps = Variable(eps)

        # print(eps.requires_grad)
        print(f"Epsilon device: {eps.device}, SD device: {std.device}, MU device: {mu.device}")
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
    For generator upsampling (keeps height and width same)
    Args:
        in_channels ():
        out_channels ():
        stride ():

    Returns:

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv4x4(in_channels, out_channels, stride=2):
    """
    For discriminator downsampling (halves height and width)
    Args:
        in_channels ():
        out_channels ():
        stride ():

    Returns:

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)


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
    def __init__(self):
        super().__init__()
        self.ca_shape = config.hyperparameters['cond_dim']
        self.gf_dim = config.hyperparameters['cond_dim'] * 8
        self.noise_shape = config.hyperparameters['latent_dim']
        self.ca = ConditionalAugmentation()
        self.fc = nn.Sequential(
            nn.Linear(self.ca_shape + self.noise_shape, self.gf_dim * 4 * 4, bias=False),
            nn.BatchNorm1d(self.gf_dim * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.up1 = UpBlock(self.gf_dim, self.gf_dim // 2)
        self.up2 = UpBlock(self.gf_dim // 2, self.gf_dim // 4)
        self.up3 = UpBlock(self.gf_dim // 4, self.gf_dim // 8)
        self.up4 = UpBlock(self.gf_dim // 8, self.gf_dim // 16)
        self.toRGB = nn.Sequential(
            conv3x3(self.gf_dim // 16, 3),
            nn.Tanh(),                                  # no batchnorm or ReLu on this layer,
        )

    def forward(self, text_embedding, noise_vec):
        """

        Args:
            text_embedding ():
            noise_vec ():

        Returns:

        """
        cond_vec, mu, logvar = self.ca(text_embedding)      # sizes: [batch, 128]
        inp = torch.cat((cond_vec, noise_vec), dim=1)     # concatenate along channel dimension
        h_code = self.fc(inp)                           # [batch, (128+100)] --> [batch, 128*8*4*4]
        h_code = h_code.view(-1, self.gf_dim, 4, 4)     # [batch, 128*8*4*4] --> [batch, 128*8, 4, 4]
        h_code = self.up1(h_code)                       # [batch, 128*8, 4, 4] --> [batch, 128*4, 8, 8]
        h_code = self.up2(h_code)                       # [batch, 128*4, 8, 8] --> [batch, 128*2, 16, 16]
        h_code = self.up3(h_code)                       # [batch, 128*2, 16, 16] --> [batch, 128*1, 32, 32]
        h_code = self.up4(h_code)                       # [batch, 128*1, 32, 32] --> [batch, 128//2, 64, 64]
        out_img = self.toRGB(h_code)                    # [batch, 128//2, 64, 64] --> [batch, 3, 64, 64]

        return None, out_img, mu, logvar


class DiscriminatorStage1(nn.Module):
    """
    The stage 1 Discriminator
    """
    def __init__(self):
        super().__init__()
        self.embedding_shape = config.hyperparameters['embedding_dim']
        self.ca_shape = config.hyperparameters['cond_dim']
        self.df_dim = config.hyperparameters['discriminator_dim']
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_shape, self.ca_shape),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.ca_shape),
        )
        self.down = nn.Sequential(
            conv4x4(3, self.df_dim),                       # [batch, 3, 64, 64] --> [batch, 64, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),            # no batchnorm after first conv
            conv4x4(self.df_dim, self.df_dim * 2),                   # [batch, 64, 32, 32] --> [batch, 128, 16, 16]
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv4x4(self.df_dim * 2, self.df_dim * 4),               # [batch, 128, 16, 16] --> [batch, 256, 8, 8]
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            conv4x4(self.df_dim * 4, self.df_dim * 8),               # [batch, 256, 8, 8] --> [batch, 512, 4, 4]
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 1x1 convolutional layer to jointly learn features across image and text
        self.joint_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 8 + self.ca_shape, out_channels=self.df_dim * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self.df_dim * 2 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, text_embedding):
        text_embedding_compressed = self.fc(text_embedding)           # [batch, 768] --> [batch, 128]
        text_embedding_compressed = text_embedding_compressed.view(text_embedding.shape[0], self.ca_shape, 1, 1)    # Reshaped to the desired shape [batch, 128, 1, 1]
        # Expanded to the target size while reusing values along the spatial dimensions
        text_embedding_compressed = text_embedding_compressed.expand(text_embedding.shape[0], self.ca_shape, 4, 4)  # [batch, 128, 1, 1] --> [batch, 128, 4, 4]

        img = self.down(img)        # [batch, 3, 64, 64] --> [batch, 512, 4, 4]
        x = torch.cat((img, text_embedding_compressed), dim=1)  # concatenated along channel axis --> [batch, (512+128), 4, 4]

        x = self.joint_conv(x)      # Jointly learn features across image and text, [batch, (512+128), 4, 4] --> [batch, 64*2, 4, 4]
        x = x.view(x.size(0), -1)   # [batch, 128 * 4 * 4]   # flatten [batch, 64*2, 4, 4] --> [batch, 128 * 4 * 4]

        output = self.fc_out(x)     # [batch, 128 * 4 * 4] --> [batch, 1]

        return output


class GeneratorStage2(nn.Module):
    """
    The stage 2 Generator
    """
    pass


class DiscriminatorStage2(nn.Module):
    """
    The stage 2 Discriminator
    """
    pass


def test():
    device = config.device
    embedding_dim = 768  # Dimensionality of text embedding (DistilBERT output size)
    conditioning_dim = 128  # Dimensionality of the conditioning vector (Ng)
    # text_embedding = torch.randn(2, embedding_dim).to(device)
    sentences = ['A quick brown fox', "Stomps over the lazy dog"]
    text_embedding = get_sentence_embeddings(sentences).to(device)
    # Initialize the conditioning network
    text_conditioning = ConditionalAugmentation(embedding_dim, conditioning_dim).to(device)

    # Generate the conditioning vector using the reparameterization trick
    conditioning_vector, _, _ = text_conditioning(text_embedding)

    print(f"Conditioning Vector Shape: {conditioning_vector.shape}")    # Expected: [2, 128]
    print("---Conditional augmentation step passed---")

    noise_vec = torch.randn(2, 100).to(device)

    gen1 = GeneratorStage1().to(device)
    _, img, mu, sig = gen1(text_embedding, noise_vec)
    print(f"Stage 1 Img device: {img.device}, MU device: {mu.device}, SD device: {sig.device}")
    print(f"Stage 1 Generated Image Shape: {img.shape}")                        # Expected: [2, 3, 64, 64]
    print("---Stage 1 Image generation step passed---")

    dis1 = DiscriminatorStage1().to(device)
    score = dis1(img, text_embedding)
    print(f"Decision scores: {score}, device: {score.device}, shape: {score.shape}")    # Expected: [2, 1]
    print("---Stage 1 Discriminator step passed---")


if __name__=="__main__":
    test()