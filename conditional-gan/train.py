
import os
import torch, torchvision
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Critic, Generator, init_weights
from utils import gradient_penalty

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4    # try using diff lrs, one for gen and one for critic
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 20
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = v2.Compose(
    [
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
        ]
)

dataset = datasets.MNIST(root='dataset/MNIST_dataset/', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)
init_weights(gen)
init_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr = LEARNING_RATE, betas = (0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr = LEARNING_RATE, betas = (0.0, 0.9))


fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)
writer_real = SummaryWriter(f"runs/mnist/real")
writer_fake  = SummaryWriter(f"runs/mnist/fake")
writer_losses = SummaryWriter(f"runs/mnist/losses")
step = 0


# Creating models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = 0
    for batch_idx, (real, labels) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}"):
        real = real.to(device)
        labels = labels.to(device)
        print(labels.size())
        num_batches += 1

        # train critic (previously referred to as discriminator)
        # here as per the paper, we need to train the critic mode
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.rand((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise, labels)
            opt_critic.zero_grad()
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            # print(real.size(), critic_real.size(), fake.size(), critic_fake.size())
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp            # penalty
            )
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        # Train Generator
        opt_gen.zero_grad()
        output = critic(fake, labels).reshape(-1)
        loss_gen = -(torch.mean(output))
        loss_gen.backward()
        opt_gen.step()

        # Accumulate losses
        epoch_g_loss += loss_gen.item()
        epoch_d_loss += loss_critic.item()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] Loss_D: {loss_critic:.4f}, Loss_G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(noise, labels)
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
    writer_losses.add_scalar('Epoch/Critic Loss', avg_d_loss, epoch)

    if (epoch + 1) % 50 == 0:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': gen.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'generator_optimizer_state_dict': opt_gen.state_dict(),
            'critic_optimizer_state_dict': opt_critic.state_dict(),
            # 'generator_loss': lossG,
            # 'critic_loss': lossD
        }
        torch.save(checkpoint, f'models/cond_wgangp_checkpoint_epoch_{epoch + 1}.pth')
        print(f"\nModel saved at models/cond_wgangp_checkpoint_epoch_{epoch + 1}.pth")

# Save final models
final_checkpoint = {
    'epoch': NUM_EPOCHS,
    'generator_state_dict': gen.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'generator_optimizer_state_dict': opt_gen.state_dict(),
    'critic_optimizer_state_dict': opt_critic.state_dict(),
    # 'generator_loss': lossG,
    # 'critic_loss': lossD
}
torch.save(final_checkpoint, 'models/cond_wgangp_checkpoint_final.pth')
print("\n\nFinal model saved at models/cond_wgangp_checkpoint_final.pth")

# Close TensorBoard writers
writer_fake.close()
writer_real.close()
writer_losses.close()


""" Usage
checkpoint = torch.load('models/cond_wgangp_checkpoint_final.pth')

# Load model states
gen.load_state_dict(checkpoint['generator_state_dict'])
critic.load_state_dict(checkpoint['critic_state_dict'])

# Load optimizer states if needed
opt_gen.load_state_dict(checkpoint['generator_optimizer_state_dict'])
opt_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])

# Get the epoch and loss information
epoch = checkpoint['epoch']
# gen_loss = checkpoint['generator_loss']
# critic_loss = checkpoint['critic_loss']
"""