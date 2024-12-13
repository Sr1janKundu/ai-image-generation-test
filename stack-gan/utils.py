"""
Some code are taken from original implementation
"""

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