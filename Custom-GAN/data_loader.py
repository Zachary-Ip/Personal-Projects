import torch
import torchvision.datasets as dsets
from torch vision import transforms

class Data_Loader():
    def __init__(self):
        self.dataset = 'pkmn'
        self.path = 'C:\Users\ipzac\Documents\Project Data\Pokemon Sprites\Clean Sprites'
        self. imsize = 96
        self. batch = 64
        self.shuf = True
        self.train = True

    def transform(self):
        transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
        return transform
    
    def loader(self):
        transforms = self.transform()
        dataset = dsets.ImageFolder(
            root = self.path
            transform = transforms
            )
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                             batchsize = self.batch,
                                             shuffle = self.shuf,
                                             numworkers = 2,
                                             drop_last = True)
        return loader