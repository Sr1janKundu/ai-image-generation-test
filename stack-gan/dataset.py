import json
import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from utils import get_sentence_embeddings
import matplotlib.pyplot as plt

import utils
from config import image_dir, all_captions_file, img_trans_stage1, img_trans_stage2, birds_img_dir, birds_caps_file

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


class Birds(Dataset):
    def __init__(self, img_root_dir, caps_file, transform=None, text_encoder = get_sentence_embeddings):
        self.img_root_dir = img_root_dir
        self.caps_file = caps_file
        self.transform = transform
        self.text_encoder = text_encoder

        self.images = []

        for folder in os.listdir(self.img_root_dir):
            folder_path = os.path.join(self.img_root_dir, folder)
            for file in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, file))

        self.captions = {}
        try:
            with open(self.caps_file, 'r') as file:
                self.captions = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {self.caps_file}")
        except KeyError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        key = os.path.basename(self.images[index]).split('.')[0]
        try:
            captions = self.captions[key]
            caption = random.choice(captions)
            caption_encoded = self.text_encoder(caption).squeeze(0)
            if caption_encoded is not None:
                return img, caption_encoded, caption
            else:
                print(f"Encoded caption is None for {key}. Skipping.")
                raise IndexError("Skipping this index due to invalid caption.")
        except KeyError:
            print(f"Caption for {key} not found.")
            raise IndexError("Skipping this index due to missing caption.")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise IndexError("Skipping this index due to an error.")


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


def get_birds_loader(img_root_dir, caps_file, transform=None, text_encoder=get_sentence_embeddings,
                     batch_size=16, shuffle=True, num_workers=8):
    """

    Args:
        img_root_dir:
        caps_file:
        transform:
        text_encoder:
        batch_size:
        shuffle:
        num_workers:

    Returns:

    """
    dataset = Birds(
        img_root_dir=img_root_dir,
        caps_file=caps_file,
        transform=transform,
        text_encoder=text_encoder,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def test():
    # train_loader, val_loader = get_flickr8k_loaders(
    #     img_dir=image_dir,
    #     captions_file=all_captions_file,
    #     transform=img_trans_stage2,
    #     text_encoder=get_sentence_embeddings,
    #     train_ratio=0.8,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4
    # )

    # loader = get_flickr8k_loader(
    #     img_dir=image_dir,
    #     captions_file=all_captions_file,
    #     transform=img_trans_stage2,
    #     text_encoder=get_sentence_embeddings,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4
    # )
    loader = get_birds_loader(
        img_root_dir=birds_img_dir,
        caps_file=birds_caps_file,
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
        img = utils.reverse_transforms(images[i])

        # Display image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{captions[i]}", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()