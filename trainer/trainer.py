import numpy as np
import functools
import torch
from utils.visualization import plot_2D


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, original_data, criterion, optimizer, config, device,
                 data_loader=None, len_epoch=None, draw_gap=1,
                 X=None):
        # super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            
        else:
            self.len_epoch = len_epoch

        print("self.len_epoch", self.len_epoch)
        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        
        print("self.log_step", self.log_step)
        self.batch_num = len(self.data_loader)
        print("self.batch_num", self.batch_num)
        # self.batch_num = functools.reduce(
        #     lambda x, y: len(x) + len(y) if isinstance(x, list) else x + len(y),
        #     self.data_loader_[self.loader_mode],
        # )
        self.warmup = int(0.05 * self.batch_num)
        self.result_list = []
        self.draw_gap = draw_gap
        self.original_data = original_data
        self.loss_value = []

        if X != None:
            self.X = X.to(device)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(0, self.config.epochs):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            # print logged informations to the screen
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            # best = False

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        loss_sum = 0.0

        for batch_idx, (to_x, from_x) in enumerate(self.data_loader):
        
            to_x, from_x = to_x.to(self.device), from_x.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(to_x, from_x)
            
            # input = torch.cat((from_x, to_x), 0)
            # input = to_x
            # output = self.model.encoder(input)
            
            # print(input.shape)
            # print(output.shape)
            
            # exit()
            # print(f"output.shape {output.shape}")
            # print(to_x.shape)
            # print(from_x.shape)

            # _X = torch.cat([from_x, to_x], dim=1)

            # loss = self.criterion(output.to(self.device), _X.to(self.device), self.batch_num)
            # output = [to_x, from_x]
            
            loss = self.criterion(output, to_x) # 为什么是 output 和 to_x 之间求criterion
            
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()

            if batch_idx % (10 * self.log_step) == 0:
                print('Train Epoch: {} {} Loss: {:.6f} {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    'warmup' if self.criterion.is_warmup() else '',
                    ),
                )
                
            if self.config.global_loss != None and batch_idx % 100 == 0:
                final_projection = self.model.project(self.original_data)
                plot_2D(final_projection.cpu().numpy(),
                          savePath=self.config.output_dir,
                          title=str(epoch) + '-' + str(batch_idx))

            if batch_idx == self.len_epoch:
                break

            # restore the procedure data of training
            if self.config.save_procedure_data and \
                len(self.result_list) <= 40 and \
                    not self.criterion.is_warmup() and \
                        batch_idx % self.draw_gap == 0:
                _z = self.model.encoder.predict(self.X).to('cpu')
                self.result_list.append((batch_idx, _z))

        # TODO Integrate store intermediate result
        # 输出的是该epoch的平均loss
        self.loss_value.append(loss_sum / len(self.data_loader))
        return None



    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
