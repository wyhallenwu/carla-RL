from source.sac_trainer import Trainer
from source.sac import config, device
from source.carlaenv import CarlaEnv
from tqdm import tqdm
from source.utility import setup_seed

if __name__ == '__main__':
    print("*" * 20)
    print(f"use device {device}")
    print("*" * 20)
    # fix seed
    setup_seed(20)
    env = CarlaEnv()
    env.client.reload_world(False)
    trainer = Trainer(env)
    for epoch_i in tqdm(range(config['epoch']), desc="Epoch"):
        trainer.train(epoch_i)
            
    # clean env
    trainer.env.exit_env()
