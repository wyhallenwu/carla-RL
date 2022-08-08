from source.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.env.client.reload_world(False)
    trainer.training_loop()
    trainer.env.exit_env()
