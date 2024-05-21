import torch
from torch import nn
from utils import AdamWScheduleFree
import torch.optim as optim
from torch.nn import functional as F
from .layers import LLS_layer, ConvBlock, LinearBlock, TrainingHook
import logging

__all__ = ["LLS_SmallConv", "DFA_SmallConv", "DRTP_SmallConv", "DFA_SmallConvMNIST", "DFA_SmallConvL", "LLS_SmallConvL",
           "DFA_SmallConvLMNIST"]

class LLS_SmallConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(LLS_SmallConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        conv_block1 = ConvBlock(in_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = LLS_layer(block=conv_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(32*4*4), reduced_set=self.reduced_set, pooling_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = LLS_layer(block=conv_block2, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(64*2*2), reduced_set=self.reduced_set, pooling_size=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = LLS_layer(block=conv_block3, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(128*2*2), reduced_set=self.reduced_set, pooling_size=2)



        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        linear_block1 = LinearBlock(128*2*2, 512)
        self.linear_block1 = LLS_layer(block=linear_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                       hidden_dim=512, reduced_set=self.reduced_set, pooling_size=512)

        self.linear = nn.Linear(512, n_classes, bias=bias)

        # Feedback matrix
        self.feedback = None

        if self.training_mode != "BP":
            if self.feedback is not None and self.training_mode != "SharedClassifier":
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
            self.linear_block1.lr_scheduler.step()
        else:
            self.lr_scheduler.step(loss_avg)
            self.conv_block1.lr_scheduler.step(loss_avg)
            self.conv_block2.lr_scheduler.step(loss_avg)
            self.conv_block3.lr_scheduler.step(loss_avg)
            self.linear_block1.lr_scheduler.step(loss_avg)

    def reset_statistics(self):
        self.conv_block1.reset_statistics()
        self.conv_block2.reset_statistics()
        self.conv_block3.reset_statistics()
        self.linear_block1.reset_statistics()

    def print_stats(self):
        logging.info(
            "Losses - L1: {0:.4f} - L2: {1:.4f} - L3: {2:.4f} - L4: {3:.4f} ".format(
                self.conv_block1.loss_avg, self.conv_block2.loss_avg, self.conv_block3.loss_avg, self.linear_block1.loss_avg))

    def optimizer_eval(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.eval()
            self.linear_block1.optimizer.eval()
            self.conv_block1.optimizer.eval()
            self.conv_block2.optimizer.eval()
            self.conv_block3.optimizer.eval()

    def optimizer_train(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.train()
            self.linear_block1.optimizer.train()
            self.conv_block1.optimizer.train()
            self.conv_block2.optimizer.train()
            self.conv_block3.optimizer.train()


    def forward(self, x, labels=None):
        training = self.training
        x = self.conv_block1(x, labels=labels, feedback=self.feedback)
        x = self.pool1(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block2(x, labels=labels, feedback=self.feedback)
        x = self.pool2(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block3(x, labels=labels, feedback=self.feedback)
        x = torch.dropout(self.pool3(x), self.dropout, training)
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
        h1_ = self.pool1(h1)
        h2 = self.conv_block2(h1_)
        h2_ = self.pool2(h2)
        h3 = self.conv_block3(h2_)
        h6_ = self.pool3(h3)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h7, h8


class LLS_SmallConvL(LLS_SmallConv):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(LLS_SmallConvL, self).__init__(in_features, out_features, bias, lr, n_classes, momentum, weight_decay, nesterov, optimizer, milestones, gamma, training_mode, lr_scheduler, patience, temperature, label_smoothing, dropout, waveform)
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        conv_block1 = ConvBlock(in_features, 96, kernel_size=3, stride=1, padding=1)
        self.conv_block1 = LLS_layer(block=conv_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(96*4*4), reduced_set=self.reduced_set, pooling_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block2 = ConvBlock(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = LLS_layer(block=conv_block2, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(192*2*2), reduced_set=self.reduced_set, pooling_size=2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_block3 = ConvBlock(192, 512, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = LLS_layer(block=conv_block3, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                     hidden_dim=int(512*2*2), reduced_set=self.reduced_set, pooling_size=2)



        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        linear_block1 = LinearBlock(512*2*2, 1024)
        self.linear_block1 = LLS_layer(block=linear_block1, lr=lr, n_classes=n_classes, momentum=momentum,
                                     weight_decay=weight_decay, nesterov=nesterov, optimizer=optimizer,
                                     milestones=milestones, gamma=gamma, training_mode=training_mode,
                                     lr_scheduler=lr_scheduler, patience=patience, temperature=temperature,
                                     label_smoothing=label_smoothing, dropout=dropout, waveform=waveform,
                                       hidden_dim=512*2*2, reduced_set=self.reduced_set, pooling_size=512*2*2)

        self.linear = nn.Linear(1024, n_classes, bias=bias)

        # Feedback matrix
        self.feedback = None

        if self.training_mode != "BP":
            if self.feedback is not None and self.training_mode != "SharedClassifier":
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
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)



class DFA_SmallConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(DFA_SmallConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        if (training_mode == "DFA") or (training_mode == "sDFA") or (training_mode == "DRTP"):
            self.y = torch.zeros(1, n_classes)
            self.y.requires_grad = False

        else:
            self.y = None

        self.conv_block1 = ConvBlock(in_features, 32, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 32, 32, 32], feedback_mode=training_mode)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv2_hook = TrainingHook(n_classes, dim_hook=[n_classes, 64, 16, 16], feedback_mode=training_mode)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv3_hook = TrainingHook(n_classes, dim_hook=[n_classes, 128, 8, 8], feedback_mode=training_mode)

        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_block1 = LinearBlock(128*2*2, 512, training_mode=training_mode)
        self.linear1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 512], feedback_mode=training_mode)

        self.linear = nn.Linear(512, n_classes, bias=bias)

    def lr_scheduler_step(self, loss_avg=None):
        pass

    def reset_statistics(self):
        pass

    def print_stats(self):
        pass

    def optimizer_eval(self):
        pass

    def optimizer_train(self):
        pass

    def update_batch_size(self, x:torch.Tensor):
        if "DFA" in self.training_mode:
            self.y = torch.zeros(x.shape[0], self.n_classes, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None


    def forward(self, x, labels=None):
        self.update_batch_size(x)
        if labels is not None:
            labels = F.one_hot(labels, num_classes=self.n_classes).float()

        training = self.training
        x = self.conv_block1(x)
        x = self.conv1_hook(x, labels , self.y)
        x = self.pool1(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block2(x)
        x = self.conv2_hook(x, labels, self.y)
        x = self.pool2(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block3(x)
        x = self.conv3_hook(x, labels, self.y)
        x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1))
        x = self.linear1_hook(x, labels, self.y)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if y_pred.requires_grad and (self.y is not None):
            self.y.data.copy_(y_pred.data)

        return y_pred

    def forward_all_layers(self, x):
        h1 = self.conv_block1(x)
        h1_ = self.pool1(h1)
        h2 = self.conv_block2(h1_)
        h2_ = self.pool2(h2)
        h3 = self.conv_block3(h2_)
        h6_ = self.pool3(h3)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h7, h8


class DFA_SmallConvL(DFA_SmallConv):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(DFA_SmallConvL, self).__init__(in_features, out_features, bias, lr, n_classes, momentum, weight_decay, nesterov, optimizer, milestones, gamma, training_mode, lr_scheduler, patience, temperature, label_smoothing, dropout, waveform)

        self.conv_block1 = ConvBlock(in_features, 96, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 96, 32, 32], feedback_mode=training_mode)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = ConvBlock(96, 192, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv2_hook = TrainingHook(n_classes, dim_hook=[n_classes, 192, 16, 16], feedback_mode=training_mode)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = ConvBlock(192, 512, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv3_hook = TrainingHook(n_classes, dim_hook=[n_classes, 512, 8, 8], feedback_mode=training_mode)

        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_block1 = LinearBlock(512 * 2 * 2, 1024, training_mode=training_mode)
        self.linear1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 1024], feedback_mode=training_mode)

        self.linear = nn.Linear(1024, n_classes, bias=bias)


class DFA_SmallConvMNIST(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(DFA_SmallConvMNIST, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        if (training_mode == "DFA") or (training_mode == "sDFA") or (training_mode == "DRTP"):
            self.y = torch.zeros(1, n_classes)
            self.y.requires_grad = False

        else:
            self.y = None

        self.conv_block1 = ConvBlock(in_features, 32, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 32, 28, 28], feedback_mode=training_mode)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv2_hook = TrainingHook(n_classes, dim_hook=[n_classes, 64, 14, 14], feedback_mode=training_mode)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv3_hook = TrainingHook(n_classes, dim_hook=[n_classes, 128, 7, 7], feedback_mode=training_mode)

        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_block1 = LinearBlock(128*2*2, 512, training_mode=training_mode)
        self.linear1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 512], feedback_mode=training_mode)

        self.linear = nn.Linear(512, n_classes, bias=bias)

    def lr_scheduler_step(self, loss_avg=None):
        pass

    def reset_statistics(self):
        pass

    def print_stats(self):
        pass

    def optimizer_eval(self):
        pass

    def optimizer_train(self):
        pass

    def update_batch_size(self, x:torch.Tensor):
        if "DFA" in self.training_mode:
            self.y = torch.zeros(x.shape[0], self.n_classes, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None


    def forward(self, x, labels=None):
        self.update_batch_size(x)
        if labels is not None:
            labels = F.one_hot(labels, num_classes=self.n_classes).float()

        training = self.training
        x = self.conv_block1(x)
        x = self.conv1_hook(x, labels , self.y)
        x = self.pool1(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block2(x)
        x = self.conv2_hook(x, labels, self.y)
        x = self.pool2(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block3(x)
        x = self.conv3_hook(x, labels, self.y)
        x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1))
        x = self.linear1_hook(x, labels, self.y)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if y_pred.requires_grad and (self.y is not None):
            self.y.data.copy_(y_pred.data)

        return y_pred

    def forward_all_layers(self, x):
        h1 = self.conv_block1(x)
        h1_ = self.pool1(h1)
        h2 = self.conv_block2(h1_)
        h2_ = self.pool2(h2)
        h3 = self.conv_block3(h2_)
        h6_ = self.pool3(h3)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h7, h8


class DFA_SmallConvLMNIST(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(DFA_SmallConvMNIST, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        if (training_mode == "DFA") or (training_mode == "sDFA") or (training_mode == "DRTP"):
            self.y = torch.zeros(1, n_classes)
            self.y.requires_grad = False

        else:
            self.y = None

        self.conv_block1 = ConvBlock(in_features, 96, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 96, 28, 28], feedback_mode=training_mode)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = ConvBlock(96, 192, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv2_hook = TrainingHook(n_classes, dim_hook=[n_classes, 192, 14, 14], feedback_mode=training_mode)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = ConvBlock(192, 512, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv3_hook = TrainingHook(n_classes, dim_hook=[n_classes, 512, 7, 7], feedback_mode=training_mode)

        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.linear_block1 = LinearBlock(512*2*2, 1024, training_mode=training_mode)
        self.linear1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 1024], feedback_mode=training_mode)

        self.linear = nn.Linear(1024, n_classes, bias=bias)

    def lr_scheduler_step(self, loss_avg=None):
        pass

    def reset_statistics(self):
        pass

    def print_stats(self):
        pass

    def optimizer_eval(self):
        pass

    def optimizer_train(self):
        pass

    def update_batch_size(self, x:torch.Tensor):
        if "DFA" in self.training_mode:
            self.y = torch.zeros(x.shape[0], self.n_classes, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None


    def forward(self, x, labels=None):
        self.update_batch_size(x)
        if labels is not None:
            labels = F.one_hot(labels, num_classes=self.n_classes).float()

        training = self.training
        x = self.conv_block1(x)
        x = self.conv1_hook(x, labels , self.y)
        x = self.pool1(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block2(x)
        x = self.conv2_hook(x, labels, self.y)
        x = self.pool2(x)
        x = torch.dropout(x, self.dropout, training)
        x = self.conv_block3(x)
        x = self.conv3_hook(x, labels, self.y)
        x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1))
        x = self.linear1_hook(x, labels, self.y)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if y_pred.requires_grad and (self.y is not None):
            self.y.data.copy_(y_pred.data)

        return y_pred

    def forward_all_layers(self, x):
        h1 = self.conv_block1(x)
        h1_ = self.pool1(h1)
        h2 = self.conv_block2(h1_)
        h2_ = self.pool2(h2)
        h3 = self.conv_block3(h2_)
        h6_ = self.pool3(h3)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h7, h8


class DRTP_SmallConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="BP",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0, waveform="cosine"):
        super(DRTP_SmallConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.reduced_set = 20
        self.training_mode = training_mode
        self.dropout = dropout
        self.waveform = waveform
        self.optimizer = None

        if (training_mode == "DFA") or (training_mode == "sDFA") or (training_mode == "DRTP"):
            self.y = torch.zeros(1, n_classes)
            self.y.requires_grad = False

        else:
            self.y = None

        self.conv_block1 = ConvBlock(in_features, 64, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 64, 32, 32], feedback_mode=training_mode)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = ConvBlock(64, 256, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv2_hook = TrainingHook(n_classes, dim_hook=[n_classes, 256, 16, 16], feedback_mode=training_mode)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, training_mode=training_mode)
        self.conv3_hook = TrainingHook(n_classes, dim_hook=[n_classes, 256, 8, 8], feedback_mode=training_mode)

        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.linear_block1 = LinearBlock(256*4*4, 1024, training_mode=training_mode)
        self.linear1_hook = TrainingHook(n_classes, dim_hook=[n_classes, 1024], feedback_mode=training_mode)

        self.linear = nn.Linear(1024, n_classes, bias=bias)

    def lr_scheduler_step(self, loss_avg=None):
        pass

    def reset_statistics(self):
        pass

    def print_stats(self):
        pass

    def optimizer_eval(self):
        pass

    def optimizer_train(self):
        pass

    def update_batch_size(self, x:torch.Tensor):
        if "DFA" in self.training_mode:
            self.y = torch.zeros(x.shape[0], self.n_classes, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None


    def forward(self, x, labels=None):
        self.update_batch_size(x)
        if labels is not None:
            labels = F.one_hot(labels, num_classes=self.n_classes).float()

        with torch.no_grad():
            training = self.training
            x = self.conv_block1(x)
            x = self.conv1_hook(x, labels , self.y)
            x = self.pool1(x)
            x = torch.dropout(x, self.dropout, training)
            x = self.conv_block2(x)
            x = self.conv2_hook(x, labels, self.y)
            x = self.pool2(x)
            x = torch.dropout(x, self.dropout, training)
            x = self.conv_block3(x)
            x = self.conv3_hook(x, labels, self.y)
            x = torch.dropout(self.pool3(x), self.dropout, training)
        x = self.linear_block1(x.view(x.size(0), -1))
        x = self.linear1_hook(x, labels, self.y)
        x = torch.dropout(x, self.dropout, training)
        y_pred = self.linear(x)

        if y_pred.requires_grad and (self.y is not None):
            self.y.data.copy_(y_pred.data)

        return y_pred

    def forward_all_layers(self, x):
        h1 = self.conv_block1(x)
        h1_ = self.pool1(h1)
        h2 = self.conv_block2(h1_)
        h2_ = self.pool2(h2)
        h3 = self.conv_block3(h2_)
        h6_ = self.pool3(h3)
        h7 = self.linear_block1(h6_.view(h6_.size(0), -1))
        h8 = self.linear(h7)

        return h1, h2, h3, h7, h8

