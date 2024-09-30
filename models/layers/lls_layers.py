import torch
from torch import nn
from utils import AdamWScheduleFree, SGDScheduleFree
import torch.optim as optim
from torch.nn import functional as F
import numpy as np

__all__ = ["LLS_layer", "LinearBlock", "ConvBlock", "ConvDWBlock"]


def generate_frequency_matrix(num_rows, num_cols, min_freq=50, max_freq=2000, freq=None):
    if freq is None:
        frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1).cuda()
    else:
        frequencies = freq
    # phases = torch.randn(num_rows, 1) * 2 * 3.14159
    t = torch.arange(num_cols).float().unsqueeze(0).cuda()
    sinusoids = torch.sin(frequencies * t )
    return sinusoids

# def generate_frequency_matrix(num_rows, num_cols, min_freq=100, max_freq=2000, freq=None):
#     frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1)
#     # phases = torch.randn(num_rows, 1) * 2 * 3.14159
#     t = torch.arange(num_cols).float().unsqueeze(0)
#     sinusoids = torch.cos(np.pi*frequencies * (t + 0.5)/num_cols)
#     return sinusoids

def compute_LocalLosses(activation, labels, local_classifier, temperature=1, label_smoothing=0.0, act_size=8):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    local_classifier_red = F.adaptive_avg_pool1d(local_classifier, latents.size(1))
    layer_pred = torch.matmul(latents, local_classifier_red.T)
    loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    return loss


def compute_LLS(activation, labels, temperature=1, label_smoothing=0.0, act_size=1, n_classes=10,
                modulation_term=None, modulation=False, freq=None, waveform="cosine"):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=512, freq=freq).cuda()
    # basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=latents.size(1) - 50).cuda()
    if waveform == "square":
        basis = torch.sign(basis)

    latents = F.normalize(latents, dim=1)
    layer_pred = torch.matmul(latents, basis.T)
    if modulation == 1:
        layer_pred = modulation_term*layer_pred
    if modulation == 2:
        layer_pred = torch.matmul(layer_pred, modulation_term)
    loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    return loss


def compute_LLS_Random(activation, labels, random_basis, temperature=1, label_smoothing=0.0, act_size=8):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    random_basis_red = F.adaptive_avg_pool1d(random_basis, latents.size(1))
    layer_pred = torch.matmul(latents, random_basis_red.T)
    loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, training_mode=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        if training_mode == "DRTP":
            self.nonlinearity = nn.Tanh()
        else:
            self.nonlinearity = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        return x

class ConvDWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvDWBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.nonl1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nonl2 = nn.LeakyReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonl1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonl2(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, training_mode=None):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)
        if training_mode == "DRTP":
            self.nonlinearity = nn.Tanh()
        else:
            self.nonlinearity = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        return x


class LLS_layer(nn.Module):
    def __init__(self, block:nn.Module, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="LLS",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0,
                 waveform="cosine", hidden_dim = 2048, reduced_set=20, pooling_size = 4, scaler = False):
        super(LLS_layer, self).__init__()
        self.block = block
        self.lr = lr
        self.n_classes = n_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.training_mode = training_mode
        self.patience = patience
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.waveform = waveform
        self.milestones = milestones
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.reduced_set = reduced_set
        self.pooling_size = pooling_size
        self.scaler = None

        if self.training_mode == "LocalLosses":
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])), requires_grad=True)
            self.training_mode = "LocalLosses"

        elif self.training_mode == "LLS_Random":
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
        elif self.training_mode == "LLS_M_Random":
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
            self.modulation = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, 1])), requires_grad=True)
        elif self.training_mode == "LLS_MxM_Random":
            self.feedback = nn.Parameter(torch.Tensor(0.1 * torch.randn([n_classes, hidden_dim])),
                                         requires_grad=False)
            self.modulation = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
        elif self.training_mode == "LLS_M":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes])), requires_grad=True)
            self.modulation_mode = 1
        elif self.training_mode == "LLS_MxM":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([n_classes, n_classes])), requires_grad=True)
            self.modulation_mode = 2
        elif self.training_mode == "LLS_MxM_reduced":
            self.feedback = nn.Parameter(torch.Tensor(0.01 * torch.randn([self.reduced_set, n_classes])), requires_grad=True)
            self.modulation_mode = 2

        # Optimizer
        if training_mode != "BP":
            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                           nesterov=nesterov)
            elif optimizer == "Adam":
                self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == "AdamWSF":
                self.optimizer = AdamWScheduleFree(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == "SGDSF":
                self.optimizer = SGDScheduleFree(self.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"{optimizer} is not supported")

            if lr_scheduler == "MultiStepLR":
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            elif lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience, verbose = True)

        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def record_statistics(self, loss, batch_size):
        self.loss_hist += loss.item() * batch_size
        self.samples += batch_size
        self.loss_avg = self.loss_hist / self.samples if self.samples > 0 else 0

    def reset_statistics(self):
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def forward(self, x, labels=None, feedback=None, x_err=None):
        training = self.training

        if self.training_mode == "BP" or not training or labels is None:
            return self.block(x)
        else:
            out = self.block(x.detach())
            if self.training_mode == "LLS":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                   self.n_classes, waveform=self.waveform)

            elif self.training_mode == "LLS_M" or self.training_mode == "LLS_MxM" or self.training_mode == "LLS_MxM_reduced":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                   self.n_classes if self.training_mode != "LLS_MxM_reduced" else self.reduced_set,
                                   modulation=self.modulation_mode, modulation_term=self.feedback,
                                   waveform=self.waveform)

            elif self.training_mode == "LLS_Random" or self.training_mode == "LLS_M_Random" or self.training_mode == "LLS_MxM_Random":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                if self.training_mode == "LLS_Random":
                    feedback = self.feedback
                elif self.training_mode == "LLS_M_Random":
                    feedback = self.modulation * self.feedback
                else:
                    feedback = torch.matmul(self.modulation, self.feedback)
                loss = compute_LocalLosses(out, labels, feedback, temperature, label_smoothing, act_size=self.pooling_size)

            elif self.training_mode == "LocalLosses":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LocalLosses(out, labels, self.feedback, temperature, label_smoothing, act_size=self.pooling_size)
            else:
                raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.record_statistics(loss.detach(), x.size(0))

            return out.detach()