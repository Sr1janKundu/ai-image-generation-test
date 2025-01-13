import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        """

        Args:
            num_timesteps ():
            beta_start ():
            beta_end ():
        """
        # initialize the parameters
        self.num_timestep = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # create betas to increase linearly from start to end
        self.betas = torch.linespace(beta_start, beta_end, num_timesteps)

        # initialize all variable needed for forward and reverse process
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        """
        The forward process
        Args:
            original (): image
            noise (): original noise sample
            t (): timestep

        Returns:

        """
        # images and noises will be of [BxCxHxW], timestep will be 1d tensor of size b
        original_shape = original.shape
        batch_size = original_shape[0]

        # reshape the following to [Bx1x1x1]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        # apply the forward process equation
        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod*original + sqrt_one_minus_alpha_cum_prod*noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Takes the image xt, and gives us a sample from our learnt reverse distribution.
        Args:
            xt ():
            noise_pred (): Noise prediction from model
            t (): timestep

        Returns:

        """
        # save original image x0 for visualizations, and get that using the following equation:
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]