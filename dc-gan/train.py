"""
Training DCGAN on MNIST and celeb dataset(s)
"""

import os
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Discriminator, Generator, init_weights


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1        # change to 1 for MNIST
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = v2.Compose(
    [
        v2.Resize(IMAGE_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
        ]
)

dataset = datasets.MNIST(root='dataset/MNIST_dataset/', train=True, transform=transforms, download=True)
# dataset = datasets.ImageFolder(root='dataset/celeb_dataset/', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
init_weights(gen)
init_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = LEARNING_RATE,betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)
writer_real = SummaryWriter(f"runs/mnist/real")
writer_fake  = SummaryWriter(f"runs/mnist/fake")
writer_losses = SummaryWriter(f"runs/mnist/losses")
# writer_real = SummaryWriter(f"runs/celeb/real")
# writer_fake  = SummaryWriter(f"runs/celeb/fake")
# writer_losses = SummaryWriter(f"runs/celeb/losses")
step = 0


# Creating models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = 0
    for batch_idx, (real, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}"):
        real = real.to(device)
        num_batches += 1
        noise = torch.rand((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        # Train discriminator
        opt_disc.zero_grad()
        disc_real = disc(real).reshape(-1)
        real_labels = torch.ones_like(disc_real).to(device)
        loss_disc_real = criterion(disc_real, real_labels)
        disc_fake = disc(fake).reshape(-1)
        fake_labels = torch.zeros_like(disc_fake).to(device)
        loss_disc_fake = criterion(disc_fake, fake_labels)
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        opt_gen.zero_grad()
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output).to(device))
        loss_gen.backward()
        opt_gen.step()

        # Accumulate losses
        epoch_g_loss += loss_gen.item()
        epoch_d_loss += loss_disc.item()
        # Calculate and log discriminator accuracy
        d_real_acc = (disc_real >= 0.5).float().mean()
        d_fake_acc = (disc_fake < 0.5).float().mean()
        d_acc = (d_real_acc + d_fake_acc) / 2
        writer_losses.add_scalar('Batch/Discriminator Accuracy', d_acc.item(), step)

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] Loss_D: {loss_disc:.4f}, Loss_G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

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
        torch.save(checkpoint, f'models/dcgan_checkpoint_epoch_{epoch + 1}.pth')
        print(f"\nModel saved at models/dcgan_checkpoint_epoch_{epoch + 1}.pth")

# Save final models
final_checkpoint = {
    'epoch': NUM_EPOCHS,
    'generator_state_dict': gen.state_dict(),
    'discriminator_state_dict': disc.state_dict(),
    'generator_optimizer_state_dict': opt_gen.state_dict(),
    'discriminator_optimizer_state_dict': opt_disc.state_dict(),
    # 'generator_loss': lossG,
    # 'discriminator_loss': lossD
}
torch.save(final_checkpoint, 'models/dcgan_checkpoint_final.pth')
print("\n\nFinal model saved at models/dcgan_checkpoint_final.pth")

# Close TensorBoard writers
writer_fake.close()
writer_real.close()
writer_losses.close()


""" Usage
checkpoint = torch.load('models/dcgan_checkpoint_final.pth')

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