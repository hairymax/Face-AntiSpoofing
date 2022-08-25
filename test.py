from src import config as cfg
from src.train_main import TrainMain

conf = cfg.get_default_config()
conf = cfg.set_job(conf)
conf.epochs = 1

trainer = TrainMain(conf)
trainer.train_model()