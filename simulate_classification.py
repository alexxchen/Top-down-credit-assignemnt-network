import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import datasets, transforms

import gymnasium as gym
from gymnasium import spaces

import time
import itertools
from lib.optimizer import flat_param, Adam, BasicSGD, update_model
from resnet import ResNet
from mlp import MLP

def param_count(model):
    current_idx = 0
    for module in model.blocks:
        layer_width = module.out_width
        current_idx += layer_width
    return current_idx

class simulator(gym.Env):
    def __init__(self, device='cpu', init_seed=0, config = None):
        self.device = device
        self.eps = np.finfo(np.float32).eps.item()

        self.config = config

        # 设置N-way分类任务的参数
        self.available_classes = config.get("class_labels")
        self.n_way = config.get("n_way", len(self.available_classes))
        
        # 确保n_way不超过可用的类别数
        assert self.n_way <= len(self.available_classes), 'n_way should smaller than available classes'
        
        self.episode = config["episode"]
        self.lr = config["lr"]
        self.L2_reg = config["L2_reg"]
        self.samples_per_class = config["samples_per_class"]

        self.init_seed = init_seed

        # 使用n_way作为输出类别数
        if config["structure"] == 'resnet':
            self.agent = ResNet(input_size=config["in_dim"], 
                             hidden_size=config["hid_dim"], 
                             num_blocks=config["num_blocks"], 
                             output_size=self.n_way, 
                             activation=config["activation"], 
                             use_norm=False,
                             initial_method=config["init_method"])
        else:
            self.agent = MLP(input_size=config["in_dim"], 
                             hidden_size=config["hid_dim"], 
                             num_blocks=config["num_blocks"], 
                             output_size=self.n_way, 
                             activation=config["activation"], 
                             initial_method=config["init_method"])

        self.agent.to(self.device)
        self.param_num = param_count(self.agent)

        params_flat, self.model_shapes = flat_param(self.agent)
        self.agent_true_param_num = len(params_flat)

        if config['dataset'] == 'MNIST':
            train_set = datasets.MNIST('MNIST_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
            test_set = datasets.MNIST('MNIST_data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        elif config['dataset'] == 'FashionMNIST':
            train_set = datasets.FashionMNIST('MNIST_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]))
            test_set = datasets.FashionMNIST('MNIST_data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]))
        elif config['dataset'] == 'EMNIST':
            train_set = datasets.EMNIST('MNIST_data', split='byclass', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
            test_set = datasets.EMNIST('MNIST_data', split='byclass', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

        # 预先加载所有类别的数据
        self._preload_data(train_set, test_set)
        
        if self.n_way < len(self.available_classes):
            self.selected_classes = self._select_random_classes(seed=0)
        else:
            self.selected_classes = self.available_classes
        
        # 创建初始数据集
        self._create_datasets_from_preloaded(seed=0)
        
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(self.param_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(2*self.n_way,), dtype=np.float64)

    def _preload_data(self, train_set, test_set):
        """预先加载所有类别的数据"""
        # 为每个类别存储训练和测试数据
        self.train_data_by_class = {}
        self.test_data_by_class = {}
        
        # 为每个可用类别加载数据
        for class_label in self.available_classes:
            # 筛选训练集
            train_indices = [i for i, (_, label) in enumerate(train_set) if label == class_label]
            train_class_data = torch.utils.data.Subset(train_set, train_indices)
            
            # 存储整个类别的数据，不进行采样
            train_loader = torch.utils.data.DataLoader(
                train_class_data, 
                batch_size=len(train_class_data), 
                shuffle=False,
                num_workers=0  # 避免多进程问题
            )
            train_inputs, train_targets = next(iter(train_loader))
            
            # 存储到字典中
            self.train_data_by_class[class_label] = (train_inputs, train_targets)
            
            # 同样处理测试集
            test_indices = [i for i, (_, label) in enumerate(test_set) if label == class_label]
            test_class_data = torch.utils.data.Subset(test_set, test_indices)
            
            # 测试集加载所有样本
            test_loader = torch.utils.data.DataLoader(
                test_class_data, 
                batch_size=len(test_class_data), 
                shuffle=False
            )
            test_inputs, test_targets = next(iter(test_loader))
            
            self.test_data_by_class[class_label] = (test_inputs, test_targets)
    
    def _select_random_classes(self, seed=0):
        """根据种子随机选择N个类别"""
        # 设置随机种子
        rng = random.Random(seed)
        return rng.sample(self.available_classes, self.n_way)
    
    def _create_datasets_from_preloaded(self, seed=0):
        """从预先加载的数据中创建训练和测试数据集，使用种子控制随机性"""
        train_inputs_list = []
        train_targets_list = []
        test_inputs_list = []
        test_targets_list = []
        
        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 创建类别映射
        class_mapping = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # 为每个选定的类别拼接数据
        for class_label in self.selected_classes:
            # 训练数据
            train_inputs, train_targets = self.train_data_by_class[class_label]
            
            # 从该类别的所有数据中随机采样
            if len(train_inputs) > self.samples_per_class:
                indices = torch.randperm(len(train_inputs))[:self.samples_per_class]
                train_inputs = train_inputs[indices]
                train_targets = train_targets[indices]
            
            train_inputs_list.append(train_inputs)
            # 重新映射标签
            remapped_targets = torch.full_like(train_targets, class_mapping[class_label])
            train_targets_list.append(remapped_targets)
            
            # 测试数据 - 使用所有测试数据
            test_inputs, test_targets = self.test_data_by_class[class_label]
            test_inputs_list.append(test_inputs)
            # 重新映射标签
            remapped_test_targets = torch.full_like(test_targets, class_mapping[class_label])
            test_targets_list.append(remapped_test_targets)
        
        # 拼接所有类别的数据
        self.inputs = torch.cat(train_inputs_list, dim=0)
        self.targets = torch.cat(train_targets_list, dim=0)
        self.test_inputs = torch.cat(test_inputs_list, dim=0)
        self.test_targets = torch.cat(test_targets_list, dim=0)
        
        # 创建one-hot编码
        self.one_hot_targets = torch.eye(self.n_way)[self.targets.reshape(-1)].to(self.device)
        self.inputs = self.inputs.view(self.inputs.size(0), -1).to(self.device)
        self.targets = self.targets.to(self.device)
        
        # 同样处理测试数据
        self.test_one_hot_targets = torch.eye(self.n_way)[self.test_targets.reshape(-1)].to(self.device)
        self.test_inputs = self.test_inputs.view(self.test_inputs.size(0), -1).to(self.device)
        self.test_targets = self.test_targets.to(self.device)

    def reset(self, seed=0, data_seed=0):   
        
        if self.n_way < len(self.available_classes):
            self.selected_classes = self._select_random_classes(seed=data_seed)
        else:
            self.selected_classes = self.available_classes

        self._create_datasets_from_preloaded(seed=data_seed)

        # 设置模型初始化种子
        torch.manual_seed(self.init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.init_seed)
        np.random.seed(self.init_seed)
        random.seed(self.init_seed)
        torch.backends.cudnn.deterministic = True
        
        self.agent.reset_parameters()
        
        self.optimizer = self.config["inner_optimizer"](self.agent.parameters(), lr=self.lr, weight_decay=self.L2_reg)

    def run_one_episode(self, controller, eval=False, data_seed=0):

        loss, acc, param_dist = [], [], []
        for i in range(self.config["meta_batch_size"]):
            l, a, p = self.run_one_task(controller, eval, data_seed*self.config["meta_batch_size"]+i)
            loss.append(l)
            acc.append(a)
            param_dist.append(p)
        return np.mean(loss), np.mean(acc), np.mean(param_dist)

    def run_one_task(self, controller, eval=False, data_seed=0):    
        # print(f"Train inputs shape: {self.inputs.shape}, Selected classes: {self.selected_classes}")
        self.reset(data_seed=data_seed)

        self.agent.train()
        for episode in range(self.episode):
            self.optimizer.zero_grad()

            pred_and_label, out, loss, acc = self.evaluate_agent(self.agent, self.inputs, self.targets, self.one_hot_targets)

            if "use_bp" in self.config and self.config["use_bp"]:
                # print('Using Backpropagation')
                loss.backward()
            else:
                gradients = controller(pred_and_label).detach()

                current_idx = 0
                for module in self.agent.blocks:
                    layer_width = module.out_width
                    if current_idx + layer_width > gradients.size(1):
                        raise RuntimeError("gradient output does not match policy parameters")
                    
                    out_gradient = gradients[:, current_idx: current_idx + layer_width]
                    module.update(out_gradient/len(out_gradient))
                    current_idx += layer_width

            total_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)

            self.optimizer.step()

        flat_model, _ = flat_param(self.agent)
        param_dist = np.linalg.norm(flat_model)
        return -loss.item(), acc, param_dist
    
    def test_agent(self):
        print(f"Test inputs shape: {self.test_inputs.shape}, Selected classes: {self.selected_classes}")
        with torch.no_grad():
            state, out, loss, acc = self.evaluate_agent(self.agent, self.test_inputs, self.test_targets, self.test_one_hot_targets)
        return -loss.item(), acc
    
    def evaluate_agent(self, model, inputs, targets, one_hot_targets):
        criterion = torch.nn.NLLLoss()

        prob = model(inputs)

        loss = criterion(torch.log(prob), targets) 

        _, predicted = prob.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()

        acc = correct/total

        pred_and_label = torch.cat((prob, one_hot_targets), dim=-1).detach()

        return pred_and_label, prob, loss, acc

if __name__ == "__main__":
    import time
    import pickle
    import os
    from evolve_mpi import Topdown
    
    forward_config = {
        "dataset": 'FashionMNIST',
        "class_labels": [0,1,2,3,4,5,6,7,8,9],  # 所有可用类别
        "n_way": 10,  # 每次任务使用4个类别
        "meta_batch_size": 1,
        "structure": 'resnet',
        "in_dim": 784,
        "hid_dim": 100,
        "num_blocks": 2,
        "init_method": 'kaiming',
        "activation": torch.nn.functional.tanh,
        "episode": 150,
        "lr": 0.01,
        "L2_reg": 0.0,
        "samples_per_class": 7000,
        "inner_optimizer": torch.optim.Adam,
        "use_bp": False
    }
    
    device = 'cuda'
    S = simulator(device, config=forward_config)  
    
    # 注意：控制器的输入维度现在是2*n_way
    controller = Topdown(S.observation_space.shape[-1], 30, S.action_space.shape[-1]).to(device)
    orig_params_flat, model_shapes = flat_param(controller)

    check_path = 'ResNet-Adam/FashionMNIST/0_1999.check'
    # check_path = 'Experiment-history/20251021-161749/check/0_1997.check'
    assert os.path.isfile(check_path), 'no such file'
    with open(check_path, 'rb') as handle:
        mu, curr_best_mu, _, _, _ = pickle.load(handle)
    update_model(curr_best_mu, controller, model_shapes, device)
    
    print(S.run_one_episode(controller, data_seed=1999))

    # 测试多个任务，使用不同的data_seed
    num_tasks = 1
    train_results, test_results = [], []
    
    for task_idx in range(num_tasks):
        print(f"\n=== Task {task_idx+1} ===")
        
        train_accs, test_accs = [], []
        train_losses, test_losses = [], []
        for i in range(5):
            S.init_seed = i
            start = time.time()
            train_loss, train_acc, _ = S.run_one_task(controller, data_seed=task_idx)
            print(f"Time: {time.time() - start:.2f}s")
            test_loss, test_acc = S.test_agent()
            print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        train_results.append([train_losses,train_accs])
        test_results.append([test_losses, test_accs])
        
        # 打印当前任务的结果
        print(f"Task {task_idx+1} - Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
        print(f"Task {task_idx+1} - Test: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    
    # 打印所有任务的平均结果
    print("\n=== Overall Results ===")
    all_train_loss = [loss for task in train_results for loss in task[0]]
    all_test_loss = [loss for task in test_results for loss in task[0]]
    all_train_acc = [acc for task in train_results for acc in task[1]]
    all_test_acc = [acc for task in test_results for acc in task[1]]
    print(all_train_acc)
    print(all_test_acc)

    print(f"All Tasks - Train loss: {np.mean(all_train_loss):.4f} ± {np.std(all_train_loss):.4f}")
    print(f"All Tasks - Test loss: {np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}")

    print(f"All Tasks - Train: {np.mean(all_train_acc):.4f} ± {np.std(all_train_acc):.4f}")
    print(f"All Tasks - Test: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}")