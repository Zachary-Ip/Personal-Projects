import os
import json
import sys
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from aegan import AEGAN
# Define global variables
BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 500

# Util function to save images.
def save_images(GAN, vec, filename): # takes in model object, a vector? and final file name
    images = GAN.generate_samples(vec) # Generates images from preset vector
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6,) # torchvision makes grid of images
    ims = ims.numpy().transpose((1,2,0))# transpose images, assuming images come out sideways or something?
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def main():
    os.makedirs("C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\generated", exist_ok=True)
    os.makedirs("C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\reconstructed", exist_ok=True)
    os.makedirs("C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\checkpoints", exist_ok=True)

    root = 'C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\Clean Sprites'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )
    X = iter(dataloader)
    test_ims1, _ = next(X)
    test_ims2, _ = next(X)
    test_ims = torch.cat((test_ims1, test_ims2), 0)
    test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6,)
    test_ims_show = test_ims_show.numpy().transpose((1,2,0))
    test_ims_show = np.array(test_ims_show*255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\reconstructed\\test_images.png")

    noise_fn = lambda x: torch.randn((x, LATENT_DIM), device=device)
    test_noise = noise_fn(36)
    gan = AEGAN(
        LATENT_DIM,
        noise_fn,
        dataloader,
        device=device,
        batch_size=BATCH_SIZE,
        )
    start = time.time()
    for i in range(EPOCHS):
        while True:
            try:
                with open("pause.json") as f:
                    pause = json.load(f)
                if pause['pause'] == 0:
                    break
                print(f"Pausing for {pause['pause']} seconds")
                time.sleep(pause["pause"])
            except (KeyError, json.decoder.JSONDecodeError, FileNotFoundError):
                break
        elapsed = int(time.time() - start)
        elapsed = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        print(f"Epoch {i+1}; Elapsed time = {elapsed}s")
        gan.train_epoch(max_steps=285)
        if (i + 1) % 50 == 0:
            torch.save(
                gan.generator.state_dict(),"C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\checkpoints\\" + f"gen.{i:05d}.pt")
        save_images(gan, test_noise,"C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\generated\\" + f"gen.{i:04d}.png")

        with torch.no_grad():
            reconstructed = gan.generator(gan.encoder(test_ims.cuda())).cpu()
        reconstructed = tv.utils.make_grid(reconstructed[:36], normalize=True, nrow=6,)
        reconstructed = reconstructed.numpy().transpose((1,2,0))
        reconstructed = np.array(reconstructed*255, dtype=np.uint8)
        reconstructed = Image.fromarray(reconstructed)
        reconstructed.save("C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\results\\reconstructed\\" + f"gen.{i:04d}.png")

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1,2,0)))
    plt.show()


if __name__ == "__main__":
    __spec__ = None
    main()