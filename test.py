from src import config
from src.train_main import TrainMain

cnf = config.get_train_config()
cnf = config.set_job(cnf)

trainer = TrainMain(cnf)
trainer.train_model()
