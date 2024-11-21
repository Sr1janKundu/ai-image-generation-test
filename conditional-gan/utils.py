import torch
import torch.nn as nn

def gradient_penalty(critic, labels, real, fake, device='cpu'):
    """

    Args:
        critic ():
        labels ():
        real ():
        fake ():
        device ():

    Returns:

    """
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)   # need one epsilon for each image
    # print(epsilon.size(), real.size(), fake.size())
    interpolated_images = epsilon * real + (1-epsilon) * fake

    # critic score calculation
    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)     # L2 norm
    grad_penalty = torch.mean((gradient_norm - 1)**2)

    return grad_penalty