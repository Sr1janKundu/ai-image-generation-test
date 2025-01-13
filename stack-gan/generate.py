import argparse
import os
import torch
import matplotlib.pyplot as plt

import config
from utils import get_sentence_embeddings, reverse_transforms
from model import GeneratorStage1, GeneratorStage2
from transformers import DistilBertTokenizer, DistilBertModel


def generate(args):
    checkpoint_path = os.path.join('checkpoints', args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    noise_vec = torch.randn(1, 100).to(config.device)
    input_caption = input("Enter the caption:\n")
    # input_caption = ["A dog", "A cat."]
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces=True)
    txt_emb_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text_embedding = get_sentence_embeddings(sentences=[input_caption],
                                             tokenizer=tokenizer,
                                             model=txt_emb_model).to(config.device)
    # text_embedding = get_sentence_embeddings(input_caption).to(config.device)
    gen1 = GeneratorStage1().to(config.device)
    gen1.load_state_dict(checkpoint['s1_generator_state_dict'])
    gen2 = GeneratorStage2(gen1).to(config.device)
    gen2.load_state_dict(checkpoint['s2_generator_state_dict'])
    # put both models on eval mode, or batchnorm will start throwing tantrums due to single input
    gen1.eval()
    gen2.eval()
    with torch.inference_mode():
        _, img_stage1, _, _ = gen1(text_embedding, noise_vec)
        img_stage11, img_stage2, _, _ = gen2(text_embedding, noise_vec)

    # fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes = axes.flatten()

    img_stage1 = reverse_transforms(img_stage1[0])
    img_stage11 = reverse_transforms(img_stage11[0])
    img_stage2 = reverse_transforms(img_stage2[0])

    axes[0].imshow(img_stage1)
    axes[0].set_title(f"(Stage 1) {input_caption}", fontsize=8)
    axes[1].imshow(img_stage11)
    axes[1].set_title(f"(Stage 1 via 2) {input_caption}", fontsize=8)
    axes[2].imshow(img_stage2)
    axes[2].set_title(f"(Stage 2) {input_caption}", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating images with StackGAN")
    parser.add_argument('--checkpoint', type=str, default='stackgan_checkpoint_epoch_675.pth', help='Name of latest checkpoint file')

    generate(parser.parse_args())