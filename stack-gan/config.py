import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters = {
    "upsampling_mode": "nearest",
    "latent_dim": 100,
    "embedding_dim": 768,
    "cond_dim": 128,
    "discriminator_dim": 64,
}