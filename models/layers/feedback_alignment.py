"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "module.py" - Definition of hooks that allow performing FA, DFA, and DRTP training.

 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

------------------------------------------------------------------------------
"""
"""
This code was adapted from DFA implementation on:
C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Fixed Random Learning Signals Allow for
Feedforward Training of Deep Neural Networks," Frontiers in Neuroscience, vol. 15, no. 629892, 2021.
doi: 10.3389/fnins.2021.629892
"""

import torch
import torch.nn as nn
from numpy import prod
import torch.nn.functional as F
__all__ = ["TrainingHook"]


class HookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, fixed_fb_bias, feedback_mode, temperature, alpha):
        if feedback_mode in ["DFA", "sDFA", "DRTP"]:
            ctx.save_for_backward(labels, y, fixed_fb_weights, fixed_fb_bias, input)
        ctx.in1 = feedback_mode
        ctx.in2 = temperature
        ctx.in3 = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        feedback_mode = ctx.in1
        temperature = ctx.in2
        alpha = ctx.in3
        if feedback_mode == "BP":
            return grad_output, None, None, None, None, None, None, None, None
        elif feedback_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None, None, None, None, None

        labels, y, fixed_fb_weights, fixed_fb_bias, input = ctx.saved_variables
        batch_size = input.size(0)
        if feedback_mode == "DFA":
            grad_output_est = (torch.softmax(y, dim=1)-labels).mm(fixed_fb_weights.view(-1, prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif feedback_mode == "sDFA":
            grad_output_est = torch.sign(torch.softmax(y, dim=1)-labels).mm(fixed_fb_weights.view(-1, prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif feedback_mode == "DRTP":
            grad_output_est = labels.mm(fixed_fb_weights.view(-1, prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        else:
            raise NameError("=== ERROR: training mode " + str(feedback_mode) + " not supported")
        grad_fb_weights = None
        grad_fb_bias = None

        return grad_output_est, None, None, grad_fb_weights, grad_fb_bias, None, None, None, None


class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim, stride=None, padding=None):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                return self.output_grad.mm(self.fixed_fb_weights)
            elif (self.layer_type == "conv"):
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad, self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, feedback_mode, temperature:float=1., alpha:float=0.):
        super(TrainingHook, self).__init__()
        self.feedback_mode = feedback_mode
        self.temperature = temperature
        self.alpha = alpha
        assert feedback_mode in ["BP", "FA", "DFA", "DRTP", "sDFA", "shallow", ], "=== ERROR: Unsupported hook training mode " + feedback_mode + "."
        self.dim_hook = dim_hook
        # Feedback weights definition (FA feedback weights are handled in the FA_wrapper class)
        if self.feedback_mode in ["DFA", "DRTP", "sDFA"]:
            self.fixed_fb_weights = nn.Parameter(1e-4*torch.rand_like(torch.Tensor(torch.Size(dim_hook))))
            self.fixed_fb_bias = nn.Parameter(1e-4*torch.rand_like(torch.Tensor(dim_hook[0])))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None
            self.fixed_fb_bias = None

        self.training_hook = HookFunction.apply

    def reset_weights(self):
        # torch.nn.init.kaiming_uniform_(self.fixed_fb_weights, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            self.fixed_fb_weights.data = 1e-4 * torch.rand_like(torch.Tensor(torch.Size(self.dim_hook))).to(self.fixed_fb_weights.data.device)
            # torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)

        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return self.training_hook(
            input, labels, y, self.fixed_fb_weights, self.fixed_fb_bias,
            self.feedback_mode if (self.feedback_mode != "FA") else "BP", self.temperature, self.alpha)

    def __repr__(self):
        repr_str = self.__class__.__name__ + "({0}, temperature={1}, alpha={2})".format(self.feedback_mode, self.temperature, self.alpha)
        return repr_str
