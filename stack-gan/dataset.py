import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from utils import get_sentence_embeddings
import matplotlib.pyplot as plt

from config import image_dir, all_captions_file, img_trans_stage1, img_trans_stage2

class Flickr8k(Dataset):
    def __init__(self, img_dir, captions_file, idx_col = 'image', cap_col = 'caption', transform=None, text_encoder = get_sentence_embeddings):
        """

        Args:
            img_dir ():
            captions_file ():
            idx_col ():
            cap_col ():
            transform ():
        """
        super().__init__()
        self.img_dir = img_dir
        self.captions_file = captions_file
        self.idx_col = idx_col
        self.cap_col = cap_col
        self.trans = transform
        self.text_encoder = text_encoder

        self.images = os.listdir(self.img_dir)
        self.captions = pd.read_csv(self.captions_file, index_col=self.idx_col)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        img = Image.open(img_path).convert('RGB')

        if self.trans is not None:
            img = self.trans(img)

        caption = random.choice(list(self.captions.loc[self.images[index]][self.cap_col]))
        caption_encoded = self.text_encoder(caption).squeeze(0)                         # [1, 768] --> [768]

        return img, caption_encoded, caption


def get_flickr8k_loaders(img_dir, captions_file, idx_col = 'image', cap_col='caption', transform=None, text_encoder=get_sentence_embeddings,
                         train_ratio=0.8, batch_size=16, shuffle=True, num_workers=8):
    """
    Create train and validation DataLoaders for the Flickr8k dataset.

    Args:
        img_dir (str): Directory containing images.
        captions_file (str): Path to the CSV file with captions.
        idx_col (str): Column in the CSV file containing image names.
        cap_col (str): Column in the CSV file containing captions.
        transform (callable, optional): Transformations to apply to images.
        text_encoder (callable, optional): Function to encode text captions.
        train_ratio (float, optional): Proportion of the dataset to use for training.
        batch_size (int, optional): Batch size for DataLoaders.
        shuffle (bool, optional): Whether to shuffle the dataset.
        num_workers (int, optional): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = Flickr8k(
        img_dir=img_dir,
        captions_file=captions_file,
        idx_col=idx_col,
        cap_col=cap_col,
        transform=transform,
        text_encoder=text_encoder
    )

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_flickr8k_loader(img_dir, captions_file, idx_col = 'image', cap_col='caption', transform=None, text_encoder=get_sentence_embeddings,
                         batch_size=16, shuffle=True, num_workers=8):
    """
    Create train and validation DataLoaders for the Flickr8k dataset.

    Args:
        img_dir (str): Directory containing images.
        captions_file (str): Path to the CSV file with captions.
        idx_col (str): Column in the CSV file containing image names.
        cap_col (str): Column in the CSV file containing captions.
        transform (callable, optional): Transformations to apply to images.
        text_encoder (callable, optional): Function to encode text captions.
        batch_size (int, optional): Batch size for DataLoaders.
        shuffle (bool, optional): Whether to shuffle the dataset.
        num_workers (int, optional): Number of subprocesses to use for data loading.

    Returns:
        dataloader with all data
    """
    dataset = Flickr8k(
        img_dir=img_dir,
        captions_file=captions_file,
        idx_col=idx_col,
        cap_col=cap_col,
        transform=transform,
        text_encoder=text_encoder
    )

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def test():
    def reverse_transforms(image_tensor):
        # Inverse normalization
        image_tensor = image_tensor * 0.5 + 0.5  # Reverse normalization

        # Convert to uint8
        image_tensor = (image_tensor * 255).clamp(0, 255).byte()

        # Convert back to a numpy array for display
        image_array = image_tensor.permute(1, 2, 0).cpu().numpy()

        return image_array

    train_loader, val_loader = get_flickr8k_loaders(
        img_dir=image_dir,
        captions_file=all_captions_file,
        transform=img_trans_stage2,
        text_encoder=get_sentence_embeddings,
        train_ratio=0.8,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    loader = get_flickr8k_loader(
        img_dir=image_dir,
        captions_file=all_captions_file,
        transform=img_trans_stage2,
        text_encoder=get_sentence_embeddings,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    # Get a batch of data
    # data_iter = iter(train_loader)
    data_iter = iter(loader)

    images, caption_embeddings, captions = next(data_iter)

    # Print caption embeddings
    print(f"Caption Embedding Matrix (shape: {caption_embeddings.shape})")
    print(caption_embeddings)

    # Plot images in a grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        # Convert image tensor to numpy array for display
        img = reverse_transforms(images[i])

        # Display image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{captions[i]}", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()