import torch
from torch import nn
from utils import AdamWScheduleFree, SGDScheduleFree
import torch.optim as optim
from .layers import LLS_layer, ConvBlock, LinearBlock
import logging

__all__ = ["LLS_VGG8", "LLS_VGG11"]

class LLS_VGG8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(LLS_VGG8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 100 # 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        conv_block1 = ConvBlock(in_features, 128, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = LLS_layer(block=conv_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=4)

        conv_block2 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = LLS_layer(block=conv_block2, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block3 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = LLS_layer(block=conv_block3, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        conv_block4 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_block4 = LLS_layer(block=conv_block4, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block5 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block5 = LLS_layer(block=conv_block5, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block6 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block6 = LLS_layer(block=conv_block6, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool4 = nn.AdaptiveAvgPool2d((2, 2))
        linear_block1 = LinearBlock(512*2*2, 1024)
        self.linear_block1 = LLS_layer(block=linear_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                       hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=1024)

        self.linear = nn.Linear(1024, n_classes, bias=bias)

        # Feedback matrix
        self.feedback = None
        if self.training_mode == "PEPITA":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 32*32*3])), requires_grad=False)
        elif self.training_mode == "LL1":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 2048])), requires_grad=True)
        elif self.training_mode == "LL1Random":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 2048])), requires_grad=False)
            self.training_mode = "LL1"
        elif self.training_mode == "LLCosineM_shared":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes])), requires_grad=True)
            self.modulation_mode = 1
        elif self.training_mode == "LLCosineMxM_shared":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
            self.modulation_mode = 2
        elif self.training_mode == "SharedClassifier":
            self.feedback = self.linear.weight

        if self.training_mode != "BP":
            if self.feedback is not None:
                params_dict = [{"params": self.linear.parameters()}, {"params": self.feedback}]
            else:
                params_dict = [{"params": self.linear.parameters()}]

            if optimizer == "SGD":
                self.optimizer = optim.SGD(params_dict, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                           nesterov=nesterov)
            elif optimizer == "Adam":
                self.optimizer = optim.Adam(params_dict, lr=lr, weight_decay=weight_decay)
            elif optimizer == "AdamWSF":
                self.optimizer = AdamWScheduleFree(params_dict, lr=lr, weight_decay=weight_decay)
            elif optimizer == "SGDSF":
                self.optimizer = SGDScheduleFree(params_dict, lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)

    def lr_scheduler_step(self, loss_avg=None):
        if loss_avg is None:
            self.lr_scheduler.step()
            self.conv_block1.lr_scheduler.step()
            self.conv_block2.lr_scheduler.step()
            self.conv_block3.lr_scheduler.step()
            self.conv_block4.lr_scheduler.step()
            self.conv_block5.lr_scheduler.step()
            self.conv_block6.lr_scheduler.step()
            self.linear_block1.lr_scheduler.step()
        else:
            self.lr_scheduler.step(loss_avg)
            self.conv_block1.lr_scheduler.step(loss_avg)
            self.conv_block2.lr_scheduler.step(loss_avg)
            self.conv_block3.lr_scheduler.step(loss_avg)
            self.conv_block4.lr_scheduler.step(loss_avg)
            self.conv_block5.lr_scheduler.step(loss_avg)
            self.conv_block6.lr_scheduler.step(loss_avg)
            self.linear_block1.lr_scheduler.step(loss_avg)

    def reset_statistics(self):
        self.conv_block1.reset_statistics()
        self.conv_block2.reset_statistics()
        self.conv_block3.reset_statistics()
        self.conv_block4.reset_statistics()
        self.conv_block5.reset_statistics()
        self.conv_block6.reset_statistics()
        self.linear_block1.reset_statistics()

    def print_stats(self):
        logging.info(
            "Losses - L1: {0:.4f} - L2: {1:.4f} - L3: {2:.4f} - L4: {3:.4f} - L5: {4:.4f} - L6: {5:.4f} - L7: {6:.4f}".format(
                self.conv_block1.loss_avg, self.conv_block2.loss_avg, self.conv_block3.loss_avg,
                self.conv_block4.loss_avg, self.conv_block5.loss_avg, self.conv_block6.loss_avg, self.linear_block1.loss_avg))

    def optimizer_eval(self):
        if isinstance(self.optimizer, AdamWScheduleFree) or isinstance(self.optimizer, SGDScheduleFree):
            self.optimizer.eval()
            self.linear_block1.optimizer.eval()
            self.conv_block1.optimizer.eval()
            self.conv_block2.optimizer.eval()
            self.conv_block3.optimizer.eval()
            self.conv_block4.optimizer.eval()
            self.conv_block5.optimizer.eval()
            self.conv_block6.optimizer.eval()

    def optimizer_train(self):
        if isinstance(self.optimizer, AdamWScheduleFree) or isinstance(self.optimizer, SGDScheduleFree):
            self.optimizer.train()
            self.linear_block1.optimizer.train()
            self.conv_block1.optimizer.train()
            self.conv_block2.optimizer.train()
            self.conv_block3.optimizer.train()
            self.conv_block4.optimizer.train()
            self.conv_block5.optimizer.train()
            self.conv_block6.optimizer.train()


    def forward(self, x, labels=None):
        training = self.training
        x = self.conv_block1(x, labels=labels, feedback=self.feedback)
        x = self.conv_block2(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool1(x), self.dropout, training)
        x = self.conv_block3(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block4(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool2(x), self.dropout, training)
        x = self.conv_block5(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.conv_block6(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool4(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1), labels=labels, feedback=self.feedback)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if self.training_mode != "BP" and training and labels is not None:
            loss = torch.nn.functional.cross_entropy(y_pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return y_pred

    def forward_all_layers(self, x):
        h1 = self.conv_block1(x)
        h2 = self.conv_block2(h1)
        h2_ = self.pool1(h2)
        h3 = self.conv_block3(h2_)
        h4 = self.conv_block4(h3)
        h4_ = self.pool2(h4)
        h5 = self.conv_block5(h4_)
        h5_ = self.pool3(h5)
        h6 = self.conv_block6(h5_)
        h6_ = self.pool4(h6)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h4, h5, h6, h7, h8


class LLS_VGG11(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(LLS_VGG11, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 100 # 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        conv_block1 = ConvBlock(in_features, 64, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = LLS_layer(block=conv_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_block2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = LLS_layer(block=conv_block2, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block3 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = LLS_layer(block=conv_block3, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        conv_block4 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_block4 = LLS_layer(block=conv_block4, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block5 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block5 = LLS_layer(block=conv_block5, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)


        conv_block6 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block6 = LLS_layer(block=conv_block6, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_block7 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block7 = LLS_layer(block=conv_block7, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        conv_block8 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block8 = LLS_layer(block=conv_block8, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2)

        self.pool5 = nn.AdaptiveAvgPool2d((8, 8))
        linear_block1 = LinearBlock(512*8*8, 4096)
        self.linear_block1 = LLS_layer(block=linear_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                       hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2048)

        linear_block2 = LinearBlock(4096, 4096)
        self.linear_block2 = LLS_layer(block=linear_block2, lr=lr, n_classes=n_classes, momentum=momentum,
                                       weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                       milestones=milestones, gamma=gamma, training_mode=training_mode,
                                       lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                       label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                       hidden_dim=2048, reduced_set=self.reduced_set, pooling_size=2048)

        self.linear = nn.Linear(4096, n_classes, bias=bias)

        # Feedback matrix
        self.feedback = None
        if self.training_mode == "PEPITA":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 32*32*3])), requires_grad=False)
        elif self.training_mode == "LL1":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 2048])), requires_grad=True)
        elif self.training_mode == "LL1Random":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 2048])), requires_grad=False)
            self.training_mode = "LL1"
        elif self.training_mode == "LLCosineM_shared":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes])), requires_grad=True)
            self.modulation_mode = 1
        elif self.training_mode == "LLCosineMxM_shared":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
            self.modulation_mode = 2
        elif self.training_mode == "SharedClassifier":
            self.feedback = self.linear.weight

        if self.training_mode != "BP":
            if self.feedback is not None:
                params_dict = [{"params": self.linear.parameters()}, {"params": self.feedback}]
            else:
                params_dict = [{"params": self.linear.parameters()}]

            if optimizer == "SGD":
                self.optimizer = optim.SGD(params_dict, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                           nesterov=nesterov)
            elif optimizer == "Adam":
                self.optimizer = optim.Adam(params_dict, lr=lr, weight_decay=weight_decay)
            elif optimizer == "AdamWSF":
                self.optimizer = AdamWScheduleFree(params_dict, lr=lr, weight_decay=weight_decay)
            elif optimizer == "SGDSF":
                self.optimizer = SGDScheduleFree(params_dict, lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)

    def lr_scheduler_step(self, loss_avg=None):
        if loss_avg is None:
            self.lr_scheduler.step()
            self.conv_block1.lr_scheduler.step()
            self.conv_block2.lr_scheduler.step()
            self.conv_block3.lr_scheduler.step()
            self.conv_block4.lr_scheduler.step()
            self.conv_block5.lr_scheduler.step()
            self.conv_block6.lr_scheduler.step()
            self.conv_block7.lr_scheduler.step()
            self.conv_block8.lr_scheduler.step()
            self.linear_block1.lr_scheduler.step()
            self.linear_block2.lr_scheduler.step()
        else:
            self.lr_scheduler.step(loss_avg)
            self.conv_block1.lr_scheduler.step(loss_avg)
            self.conv_block2.lr_scheduler.step(loss_avg)
            self.conv_block3.lr_scheduler.step(loss_avg)
            self.conv_block4.lr_scheduler.step(loss_avg)
            self.conv_block5.lr_scheduler.step(loss_avg)
            self.conv_block6.lr_scheduler.step(loss_avg)
            self.conv_block7.lr_scheduler.step(loss_avg)
            self.conv_block8.lr_scheduler.step(loss_avg)
            self.linear_block1.lr_scheduler.step(loss_avg)
            self.linear_block2.lr_scheduler.step(loss_avg)

    def reset_statistics(self):
        self.conv_block1.reset_statistics()
        self.conv_block2.reset_statistics()
        self.conv_block3.reset_statistics()
        self.conv_block4.reset_statistics()
        self.conv_block5.reset_statistics()
        self.conv_block6.reset_statistics()
        self.conv_block7.reset_statistics()
        self.conv_block8.reset_statistics()
        self.linear_block1.reset_statistics()
        self.linear_block2.reset_statistics()

    def print_stats(self):
        logging.info(
            "Losses - L1: {0:.4f} - L2: {1:.4f} - L3: {2:.4f} - L4: {3:.4f} - L5: {4:.4f} - L6: {5:.4f} - L7: {6:.4f}".format(
                self.conv_block1.loss_avg, self.conv_block2.loss_avg, self.conv_block3.loss_avg,
                self.conv_block4.loss_avg, self.conv_block5.loss_avg, self.conv_block6.loss_avg, self.linear_block1.loss_avg))

    def optimizer_eval(self):
        if isinstance(self.optimizer, AdamWScheduleFree) or isinstance(self.optimizer, SGDScheduleFree):
            self.optimizer.eval()
            self.linear_block1.optimizer.eval()
            self.linear_block2.optimizer.eval()
            self.conv_block1.optimizer.eval()
            self.conv_block2.optimizer.eval()
            self.conv_block3.optimizer.eval()
            self.conv_block4.optimizer.eval()
            self.conv_block5.optimizer.eval()
            self.conv_block6.optimizer.eval()
            self.conv_block7.optimizer.eval()
            self.conv_block8.optimizer.eval()

    def optimizer_train(self):
        if isinstance(self.optimizer, AdamWScheduleFree) or isinstance(self.optimizer, SGDScheduleFree):
            self.optimizer.train()
            self.linear_block1.optimizer.train()
            self.linear_block2.optimizer.train()
            self.conv_block1.optimizer.train()
            self.conv_block2.optimizer.train()
            self.conv_block3.optimizer.train()
            self.conv_block4.optimizer.train()
            self.conv_block5.optimizer.train()
            self.conv_block6.optimizer.train()
            self.conv_block7.optimizer.train()
            self.conv_block8.optimizer.train()


    def forward(self, x, labels=None):
        training = self.training
        x = self.conv_block1(x, labels=labels, feedback=self.feedback)
        x = self.pool1(x)
        x = self.conv_block2(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool2(x), self.dropout, training)
        x = self.conv_block3(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block4(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.conv_block5(x, labels=labels, feedback=self.feedback)
        x = self.conv_block6(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool4(x), self.dropout, training)
        x = self.conv_block7(x, labels=labels, feedback=self.feedback)
        x = self.conv_block8(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool5(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1), labels=labels, feedback=self.feedback)
        x = self.linear_block2(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if self.training_mode != "BP" and training and labels is not None:
            loss = torch.nn.functional.cross_entropy(y_pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return y_pred

    def forward_all_layers(self, x):

        return x, x