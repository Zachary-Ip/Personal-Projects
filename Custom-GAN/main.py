from trainer import Trainer
from data_loader  import Data_Loader
from torch.backends import cudnn
from utils import make_folder

def main():
    # For fast training
    cudnn.benchmark = True
    
    # Data loader
    data_loader = Data_Loader
    
    # Create relevant directories
    make_folder() # save path
    make_folder() # sample path
    make_folder() #log path
    make_folder() #attn path
    
    
    # Train model
    trainer = Trainer(data_loader.loader())
    trainer.train()

if __name__ == '__main__':
    __spec__ = None
    main()