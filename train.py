from src import config
from src.train_main import TrainMain

cnf = config.get_train_config(spoof_categories=[[0],[1,2,3],[7,8,9]],
                              class_balancing='down')
print(cnf.num_classes)
cnf = config.set_train_job(cnf, 'AntiSpoofing_print_replay_cb-down')

trainer = TrainMain(cnf)
trainer.train_model()