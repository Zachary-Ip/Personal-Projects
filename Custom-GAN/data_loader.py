import torch
import torchvision.datasets as dsets
from torchvision import transforms

class Data_Loader():
    def __init__(self):
        self.dataset = 'pkmn'
        self.path = 'C:\\Users\\ipzac\Documents\\Project Data\\Pokemon Sprites\\Crop Sprites'
        self.imsize = 64
        self.batch = 64
        self.shuf = True
        self.train = True

    def transform(self):
        transform = transforms.Compose([
            transforms.RandomAdjustSharpness(2),
            transforms.RandomAdutoContrast(),
            transforms.RandomAffine(degrees=(0,60), translate=(5/96, 5/96), fill=(255,255,255)),
            transforms.ColorJitter(hue=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale = 0.5, p = 0.85),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
        return transform
    
    def loader(self):
        transforms = self.transform()
        dataset = dsets.ImageFolder(
            root = self.path,
            transform = transforms
            )
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                             batch_size = self.batch,
                                             shuffle = self.shuf,
                                             num_workers = 2,
                                             drop_last = True)
        return loader