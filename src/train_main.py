# @hairymax

import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.NN import MultiFTNet
from src.dataset_loader import get_train_valid_loader
from datetime import datetime


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.step = 0
        self.val_step = 0
        self.start_epoch = 0
        self.train_loader, self.valid_loader = get_train_valid_loader(self.conf)
        self.board_train_every = len(self.train_loader) // conf.board_loss_per_epoch
        self.board_valid_every = len(self.valid_loader) // conf.board_loss_per_epoch

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        run_loss = 0.
        run_acc = 0.
        run_loss_cls = 0.
        run_loss_ft = 0.
        run_val_acc = 0.
        run_val_loss_cls = 0.
        
        is_first = True

        print('Board train loss every {} steps'.format(self.board_train_every))
        print('Board valid loss every {} steps'.format(self.board_valid_every))
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False

            # Training
            print('Epoch {} started. lr: {}'.format(e, self.schedule_lr.get_last_lr()))
            self.model.train()
            print('Training on {} batches.'.format(len(self.train_loader)))
            for sample, ft_sample, labels in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                run_loss += loss
                run_acc += acc
                run_loss_cls += loss_cls
                run_loss_ft += loss_ft

                self.step += 1

                if self.step % self.board_train_every == 0 and self.step != 0:
                    board_step = self.step // self.board_train_every
                    self.writer.add_scalar('Loss/train', run_loss / self.board_train_every, board_step)
                    self.writer.add_scalar('Acc/train', run_acc / self.board_train_every, board_step)
                    self.writer.add_scalar('Loss_cls/train', run_loss_cls / self.board_train_every, board_step)
                    self.writer.add_scalar('Loss_ft/train', run_loss_ft / self.board_train_every, board_step)
                    self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], board_step)

                    run_loss = 0.
                    run_acc = 0.
                    run_loss_cls = 0.
                    run_loss_ft = 0.
                    
            self.schedule_lr.step()

            # Validation
            self.model.eval()
            print('Validation on {} batches.'.format(len(self.valid_loader)))
            for sample, labels in tqdm(iter(self.valid_loader)):
                with torch.no_grad():
                    acc, loss_cls = self._valid_batch_data(sample, labels)
                run_val_acc += acc
                run_val_loss_cls += loss_cls

                self.val_step += 1

                if self.val_step % self.board_valid_every == 0 and self.val_step != 0:
                    board_step = self.val_step // self.board_valid_every
                    self.writer.add_scalar('Acc/valid', run_val_acc / self.board_valid_every, board_step)
                    self.writer.add_scalar('Loss_cls/valid', run_val_loss_cls / self.board_valid_every, board_step)
                    run_val_acc = 0.
                    run_val_loss_cls = 0.
            
            self._save_state('epoch-{}'.format(e))
        
        self.writer.close()


    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()


    def _valid_batch_data(self, img, labels):
        labels = labels.to(self.conf.device)
        embeddings = self.model.forward(img.to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        acc = self._get_accuracy(embeddings, labels)[0]

        return acc, loss_cls.item()

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model)#self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, stage):
        save_path = self.conf.model_path
        job_name = self.conf.job_name
        time_stamp = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_{}.pth'.format(time_stamp, job_name, stage)))
