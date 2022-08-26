from src import config as cfg
from src.train_main import TrainMain

cnf = cfg.get_train_config()
cnf = cfg.set_train_job(cnf, 'AntiSpoofing_bin')

trainer = TrainMain(cnf)
trainer.train_model()