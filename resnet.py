import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, use_norm=False):
        super(ResNetBlock, self).__init__()
        assert in_features == out_features, "ResNet block requires same input and output dimensions"
        
        self.use_norm = use_norm
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        
        # 添加批归一化层
        if self.use_norm:
            self.bn1 = nn.BatchNorm1d(out_features)
            self.bn2 = nn.BatchNorm1d(out_features)

        self.out_width = out_features
        self.last_output = None
        self.activation = activation

    def forward(self, x):
        identity = x  # 残差连接的恒等映射
        
        # 第一个全连接层
        out = self.fc1(x)
        
        # 如果使用批归一化，在激活函数前应用
        if self.use_norm:
            out = self.bn1(out)
        
        out = self.activation(out)  
        # out = self.dropout1(out)
        
        # 第二个全连接层
        out = self.fc2(out)
        
        # 如果使用批归一化，在残差连接前应用
        if self.use_norm:
            out = self.bn2(out)
            # out = self.dropout2(out)
        
        # 添加残差连接并应用激活函数
        out = self.activation(out + identity)
        self.last_output = out  # 保存输出，但不截断计算图
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

class ResNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size, activation, use_norm=False, initial_method='kaiming'):
        super(ResNet, self).__init__()
        self.input_size = input_size
        
        self.initial_method = initial_method
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        self.blocks.append(input_block(input_size, hidden_size, activation))

        for _ in range(num_blocks):
            self.blocks.append(ResNetBlock(hidden_size, hidden_size, activation, use_norm=use_norm))

        self.blocks.append(output_block(hidden_size, output_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.initial_method == 'zero':
            weight_init = nn.init.constant_
            for block in self.blocks:
                if isinstance(block, ResNetBlock):
                    weight_init(block.fc1.weight, 0)
                    weight_init(block.fc2.weight, 0)
                else:
                    weight_init(block.fc.weight, 0)

        elif self.initial_method == 'xavier':
            weight_init = nn.init.xavier_uniform_
            for block in self.blocks:
                if isinstance(block, ResNetBlock):
                    weight_init(block.fc1.weight)
                    weight_init(block.fc2.weight)
                else:
                    weight_init(block.fc.weight)
        
        elif self.initial_method == 'ortho':
            weight_init = nn.init.orthogonal_
            for block in self.blocks:
                if isinstance(block, ResNetBlock):
                    weight_init(block.fc1.weight)
                    weight_init(block.fc2.weight)
                else:
                    weight_init(block.fc.weight)

        elif self.initial_method == 'kaiming':
            weight_init = nn.init.kaiming_uniform_
            for block in self.blocks:
                if isinstance(block, ResNetBlock):
                    weight_init(block.fc1.weight, a=math.sqrt(5))
                    weight_init(block.fc2.weight, a=math.sqrt(5))
                else:
                    weight_init(block.fc.weight, a=math.sqrt(5))
        else:
            assert 1==0, 'no such initialzation method'

        for block in self.blocks:
            if isinstance(block, ResNetBlock):
                nn.init.constant_(block.fc1.bias, 0)
                nn.init.constant_(block.fc2.bias, 0)
            else:
                nn.init.constant_(block.fc.bias, 0)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
