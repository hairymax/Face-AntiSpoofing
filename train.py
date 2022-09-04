from src.config import TrainConfig
from src.train_main import TrainMain
cnf = TrainConfig(spoof_categories=[[0],[1,2,3],[7,8,9]])
cnf.set_job('print_replay')

print("Number of classes:", cnf.num_classes)
print("Device:", cnf.device)
trainer = TrainMain(cnf)
trainer.train_model()
