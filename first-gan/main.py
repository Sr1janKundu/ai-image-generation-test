import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Discriminator(nn.Module):
    """
    A simple discriminator class
    """
    def __init__(self, img_dim):
        """

        Args:
            img_dim (int): Image size (eg, 784 for MNIST)
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(      ## simple discriminator network
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    """
    A simple Generator class
    """
    def __init__(self, z_dm, img_dim):
        """

        Args:
            z_dm (int): dimension of latent noise
            img_dim (int): Image size, same as img_dim of Discriminator class
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(       ## simple generator network
            nn.Linear(z_dm, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),      # to normalize the output between -1 and 1, keeping same like normalized input for discriminator
        )

    def forward(self, x):
        return self.gen(x)

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4       # cuz Karpathy twitted "3e-4 is the best learning rate for Adam, hands down."
z_dim = 64      # experiment with 128, 256
image_dim = 28*28*1
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)
transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Normalize((0.1307,), (0.3081,)),
        v2.ToDtype(torch.float32, scale=True)
    ]
)
dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/first_gan/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/first_gan/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in tqdm(enumerate(loader)):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train discriminator: max[log(D(real)) + log(1-D(G(z))] or min[-{log(D(real)) + log(1-D(G(z))}]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))