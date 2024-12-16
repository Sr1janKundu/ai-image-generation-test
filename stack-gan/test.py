import torch
from model import DiscriminatorStage2  # Replace with your model
from torch.optim import Adam

# Path to your checkpoint
checkpoint_path = 'checkpoints/stackgan_checkpoint_epoch_20.pth'

# Initialize the model and optimizer
model = DiscriminatorStage2()
optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Match your optimizer setup

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Check optimizer state_dict
print("Checkpoint optimizer parameter groups:")
for i, group in enumerate(checkpoint['s2_discriminator_optimizer_state_dict']['param_groups']):
    print(f"Group {i}: {group.keys()}")

print("\nCurrent optimizer parameter groups:")
for i, group in enumerate(optimizer.state_dict()['param_groups']):
    print(f"Group {i}: {group.keys()}")

# Align the parameter groups if they don't match
try:
    optimizer.load_state_dict(checkpoint['s2_discriminator_optimizer_state_dict'])
    print("Optimizer state loaded successfully.")
except ValueError as e:
    print(f"Optimizer state loading failed: {e}")

    # Align parameter groups manually
    checkpoint_state = checkpoint['s2_discriminator_optimizer_state_dict']
    current_state = optimizer.state_dict()

    # Ensure the parameter groups match
    checkpoint_state['param_groups'] = current_state['param_groups']

    # Load the adjusted state
    optimizer.load_state_dict(checkpoint_state)
    print("Optimizer state loaded after alignment.")
