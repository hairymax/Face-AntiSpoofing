from src import config as cfg
from src.train_main import TrainMain

cnf = cfg.get_default_config()
cnf = cfg.set_job(cnf)

trainer = TrainMain(cnf)
trainer.train_model()