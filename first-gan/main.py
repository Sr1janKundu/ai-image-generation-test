import os
import torch, torchvision
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
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    """
    A simple Generator class
    """
    def __init__(self, z_dim, img_dim):
        """

        Args:
            z_dim (int): dimension of latent noise
            img_dim (int): Image size, same as img_dim of Discriminator class
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(       ## simple generator network
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),      # to normalize the output between -1 and 1, keeping same like normalized input for discriminator
        )

    def forward(self, x):
        return self.gen(x)

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4       # cuz Karpathy twitted "3e-4 is the best learning rate for Adam, hands down."
z_dim = 64      # experiment with 128, 256
image_dim = 28*28*1     #784
batch_size = 32
num_epochs = 500

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
        # using MNIST mean and sd as `v2.Normalize((0.1307,), (0.3081,))` will not work as:
        # nn.Tanh() output of the Generator is (-1, 1)
        # MNIST values are [0, 1]
        # Normalize does the following for each channel: image = (image - mean) / std
        # So v2.Normalize((0.5,), (0.5,)) converts [0, 1] to [-1, 1], which is ALMOST correct,
        # because nn.Tanh() output of Generator (-1, 1) excluding one and minus one.
        # v2.Normalize((0.1307,), (0.3081,)) converts [0, 1] to ≈ (-0.42, 2.82).
        # But Generator can not generate values greater than 0.9999... ≈ 1, so it will not generate 2.8 for white color.
        # That is why transforms.Normalize((0.1307,), (0.3081,)) will not work.
        # To use transforms.Normalize((0.1307,), (0.3081,)) you should multiply nn.Tanh() with 2.83 ≈ nn.Tanh() * 2.83 ≈ (-2.83, 2.83)
    ]
)

dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/fake")
writer_real = SummaryWriter(f"runs/real")
writer_losses = SummaryWriter(f"runs/losses")
step = 0

# Creating models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

for epoch in range(num_epochs):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = 0

    for batch_idx, (real, _) in tqdm(enumerate(loader), total=len(loader)):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        num_batches += 1

        # Train discriminator: max[log(D(real)) + log(1-D(G(z))] or min[-{log(D(real)) + log(1-D(G(z))}]
        opt_disc.zero_grad()
        disc_real = disc(real).view(-1)
        real_labels = torch.ones_like(disc_real).to(device)     # Label for real images is 1
        lossD_real = criterion(disc_real, real_labels)
        # Generate latent noise to generate fake images
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_fake = disc(fake).view(-1)
        fake_labels = torch.zeros_like(disc_fake).to(device)    # Label for fake images is 0
        lossD_fake = criterion(disc_fake, fake_labels)
        lossD = (lossD_real + lossD_fake) / 2
        # disc.zero_grad()                            # ??
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: min[log(1-D(G(z)) <-> max[log(D(G(z))]
        # the first expression leads to saturated/weak gradients; in practice, better to use second expression
        # now we want to use `fake = gen(noise)` from previous calculation, without calculating again,
        # but calling disc.zero_grad() clears the cache, so we can do `disc(fake.detach()).view(-1)` at line 94 instead of `disc(fake).view(-1)`, or
        # we can do lossD.backward(retain_graph=True) at line 98

        opt_gen.zero_grad()
        output = disc(fake).view(-1)
        lossG = criterion(output, real_labels)
        # gen.zero_grad()                             # ??
        lossG.backward()
        opt_gen.step()

        # Accumulate losses
        epoch_g_loss += lossG.item()
        epoch_d_loss += lossD.item()
        # Calculate and log discriminator accuracy
        d_real_acc = (disc_real >= 0.5).float().mean()
        d_fake_acc = (disc_fake < 0.5).float().mean()
        d_acc = (d_real_acc + d_fake_acc) / 2
        writer_losses.add_scalar('Batch/Discriminator Accuracy', d_acc.item(), step)

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {lossD:.4f}, Loss_G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )
                step += 1

    # Log average losses for the epoch
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches
    writer_losses.add_scalar('Epoch/Generator Loss', avg_g_loss, epoch)
    writer_losses.add_scalar('Epoch/Discriminator Loss', avg_d_loss, epoch)

    if (epoch + 1) % 50 == 0:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': gen.state_dict(),
            'discriminator_state_dict': disc.state_dict(),
            'generator_optimizer_state_dict': opt_gen.state_dict(),
            'discriminator_optimizer_state_dict': opt_disc.state_dict(),
            # 'generator_loss': lossG,
            # 'discriminator_loss': lossD
        }
        torch.save(checkpoint, f'models/gan_checkpoint_epoch_{epoch + 1}.pth')
        print(f"\nModel saved at models/gan_checkpoint_epoch_{epoch + 1}.pth")

# Save final models
final_checkpoint = {
    'epoch': num_epochs,
    'generator_state_dict': gen.state_dict(),
    'discriminator_state_dict': disc.state_dict(),
    'generator_optimizer_state_dict': opt_gen.state_dict(),
    'discriminator_optimizer_state_dict': opt_disc.state_dict(),
    # 'generator_loss': lossG,
    # 'discriminator_loss': lossD
}
torch.save(final_checkpoint, 'models/gan_checkpoint_final.pth')
print("\n\nFinal model saved at models/gan_checkpoint_final.pth")


"""
checkpoint = torch.load('models/gan_checkpoint_final.pth')

# Load model states
gen.load_state_dict(checkpoint['generator_state_dict'])
disc.load_state_dict(checkpoint['discriminator_state_dict'])

# Load optimizer states if needed
opt_gen.load_state_dict(checkpoint['generator_optimizer_state_dict'])
opt_disc.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

# Get the epoch and loss information
epoch = checkpoint['epoch']
# gen_loss = checkpoint['generator_loss']
# disc_loss = checkpoint['discriminator_loss']
"""