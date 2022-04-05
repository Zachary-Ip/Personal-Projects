
from parameter import *
from trainer import Trainer
# from tester import Tester
#from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision as tv

# Define global variables
BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 500


def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    root = 'C:\\Users\\ipzac\\Documents\\Project Data\\Pokemon Sprites\\Clean Sprites'
    transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(0, translate=(5/96, 5/96), fill=(255,255,255)),
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root#,
            #transform=transform
            )
    data_loader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader, config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader, config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    __spec__ = None
    config = get_parameters()
    print(config)
    main(config)