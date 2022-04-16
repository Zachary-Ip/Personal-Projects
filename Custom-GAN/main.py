from trainer import Trainer
from data_loader  import Data_Loader
from torch.backends import cudnn
from utils import make_folder

def main():
    # For fast training
    cudnn.benchmark = True
    
    # Data loader
    data_loader = Data_Loader()
    
    # Create relevant directories
    make_folder('C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/Custom/models') # save path
    make_folder('C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/Custom/samples') # sample path
    make_folder('C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/Custom/logs') #log path
    make_folder('C:/Users/ipzac/Documents/Project Data/Pokemon Sprites/Custom/attn') #attn path
    
    
    # Train model
    trainer = Trainer(data_loader.loader())
    trainer.train()

if __name__ == '__main__':
    __spec__ = None
    main()