import os
from parameters import *
from trainer import Trainer
from data_loader import Data_Loader
from torch.backends import cudnn


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def main(config):
    # if input size remains unchanged then
    # enabling it will leads to faster runtime;
    cudnn.benchmark = True


    # Data loader
    data_loader = Data_Loader(config.dataset, config.image_path, config.imsize,
                              config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)

    trainer = Trainer(data_loader.loader(), config)
    trainer.train()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
