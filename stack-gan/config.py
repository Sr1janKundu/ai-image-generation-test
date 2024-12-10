import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters = {
    "upsampling_mode": "nearest",
    "latent_dim": 100,
    "embedding_dim": 768,
    "cond_dim": 128,
    "discriminator_dim": 64,
    "stage2_gen_res_count": 4,      # two residual blocks for 128x128 models, 4 for 256x256 blocks
}