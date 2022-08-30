from src import config
from src.train_main import TrainMain
cnf = config.get_train_config(spoof_categories='binary',#[[0],[1,2,3],[7,8,9]],
                              class_balancing='down')

cnf = config.set_train_job(cnf, 'bin_cb-down')

print("Number of classes:", cnf.num_classes)
print("Device:", cnf.device)
trainer = TrainMain(cnf)
trainer.train_model()
