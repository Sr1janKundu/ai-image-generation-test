import torch
from torchvision.transforms import v2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir = '/home/srijan/Desktop/Srijan-files/seq2seq-demo/image_captioning/Flickr_8k_Images_Captions/flickr8k/images/'
all_captions_file = '/home/srijan/Desktop/Srijan-files/seq2seq-demo/image_captioning/Flickr_8k_Images_Captions/flickr8k/captions.txt'

load_stage1 = False
load_stage2 = False

hyperparameters = {
    "upsampling_mode": "nearest",
    "latent_dim": 100,
    "embedding_dim": 768,
    "cond_dim": 128,
    "discriminator_dim": 64,
    "stage2_gen_res_count": 4,      # two residual blocks for 128x128 models, 4 for 256x256 blocks
    "gen_loss_kld_reg_param": 1,    # lambda
    "train_stage1_for": 25,         # number of epochs to train stage 1 gan first (do 600 to follow paper)
    "start_lr": 2e-4,
    "batch_size_stage1": 64,
    "batch_size_stage2": 64,

}

img_trans_stage1 = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img_trans_stage2 = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])