import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class input_block(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(input_block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.out_width = out_features
        self.last_output = None
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.fc(x))
        self.last_output = out  # 保存输出，但不截断计算图
        return out
    
    def update(self, out_gradient):
        # 直接使用已有的计算图计算梯度
        gradients = autograd.grad(
            outputs=self.last_output,
            inputs=list(self.parameters()),
            grad_outputs=out_gradient,
            retain_graph=False,
            create_graph=False,
            only_inputs=True
        )
        
        # 将计算得到的梯度设置到参数上
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad


class Block(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.out_width = out_features
        self.last_output = None
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.fc(x))
        self.last_output = out
        return out
    
    def update(self, out_gradient):
        # 直接使用已有的计算图计算梯度
        gradients = autograd.grad(
            outputs=self.last_output,
            inputs=list(self.parameters()),
            grad_outputs=out_gradient,
            retain_graph=False,  # 不保留计算图，节省内存
            create_graph=False,
            only_inputs=True
        )
        
        # 将计算得到的梯度设置到参数上
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad  # 直接赋值，而不是累加
        
class output_block(nn.Module):
    def __init__(self, in_features, out_features):
        super(output_block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.out_width = out_features
        self.last_output = None

    def forward(self, x):
        out = F.softmax(self.fc(x), dim=-1)
        self.last_output = out  # 保存输出，但不截断计算图
        return out

    def update(self, out_gradient):
        # 直接使用已有的计算图计算梯度
        gradients = autograd.grad(
            outputs=self.last_output,
            inputs=list(self.parameters()),
            grad_outputs=out_gradient,
            retain_graph=False,
            create_graph=False,
            only_inputs=True
        )
        
        # 将计算得到的梯度设置到参数上
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size, activation, initial_method='zero'):
        super(MLP, self).__init__()
        
        self.initial_method = initial_method
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        self.blocks.append(input_block(input_size, hidden_size, activation))

        for _ in range(num_blocks):
            self.blocks.append(Block(hidden_size, hidden_size, activation))

        self.blocks.append(output_block(hidden_size, output_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.initial_method == 'zero':
            weight_init = nn.init.constant_
            for block in self.blocks:
                weight_init(block.fc.weight, 0)

        elif self.initial_method == 'xavier':
            weight_init = nn.init.xavier_uniform_
            for block in self.blocks:
                weight_init(block.fc.weight)
        
        elif self.initial_method == 'ortho':
            weight_init = nn.init.orthogonal_
            for block in self.blocks:
                weight_init(block.fc.weight)
                
        elif self.initial_method == 'kaiming':
            weight_init = nn.init.kaiming_uniform_
            for block in self.blocks:
                weight_init(block.fc.weight, a=math.sqrt(5))
        else:
            assert 1==0, 'no such initialzation method'

        for block in self.blocks:
            nn.init.constant_(block.fc.bias, 0)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
