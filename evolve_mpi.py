import os
import numpy as np
import math
from mpi4py import MPI

import torch
import torch.nn as nn
import pickle
import time

from lib.es import PEPG, CMAES
from lib.optimizer import flat_param, update_model
from simulate_classification import simulator

def layer_init(layer, std=np.sqrt(2), bias_const=0.0, init='xavier'):
    if init == 'orth':
        torch.nn.init.orthogonal_(layer.weight, std)
    else:
        nn.init.xavier_uniform_(layer.weight)

    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Topdown(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, rnn_type='gru'):
        super(Topdown, self).__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.Tanh(),
            # Better performance on cartpole using std=0.1
            layer_init(nn.Linear(hid_dim, output_dim), std=0.01),
        )
        # self.actor = nn.Sequential(
        #     nn.Linear(input_dim, hid_dim),
        #     nn.Tanh(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.Tanh(),
        #     # nn.Linear(hid_dim, hid_dim),
        #     # nn.Tanh(),
        #     # Better performance on cartpole using std=0.1
        #     nn.Linear(hid_dim, output_dim),
        # )

    def forward(self, x):
        out = self.actor(x)
        return out

class Evolution():
    def __init__(self, forward_config, pgpe_params, population, hiddem_width, worker_num, generation, device, eval_interval=50):
        self.eval_interval = eval_interval
        self.device = device

        self.parallel_simulator = parallel_simulation(forward_config, pgpe_params['popsize'], hiddem_width, self.device, num_worker=worker_num)


        pgpe_params['num_params'] = self.parallel_simulator.NPARAMS
        print('Param to evolve:', self.parallel_simulator.NPARAMS)

        self.generation = generation

        self.es = PEPG(**pgpe_params)

        self.es.mu = self.parallel_simulator.orig_params_flat
        self.es.optimizer.param = self.parallel_simulator.orig_params_flat
        self.start_point = self.parallel_simulator.orig_params_flat

    def evolve_model(self, gen):

        solutions = self.es.ask()
        start = time.time()
        reward, success, path_len = self.parallel_simulator.batch_simulation(solutions, gen)

        end = time.time()
        index = reward.argmax()
        best_raw_reward = reward[index]
        worse_raw_reward = reward.min()

        last_mu = self.es.mu
        mu_grad, mu_step, sigma_change_ratio = self.es.tell(reward)

        # if gen % self.eval_interval == 0:
        #     eval_loss, eval_acc, test_loss, test_acc = self.parallel_simulator.eval(self.es.mu)
        
        from_start = np.linalg.norm(self.es.mu - self.start_point)
        from_last = np.linalg.norm(self.es.mu - last_mu)
        print('run time:', end-start)
        print('forward path length', path_len.max(), path_len.min(), path_len[index])
        print('feedback mu lenth, best lenth:', np.linalg.norm(last_mu), np.linalg.norm(self.es.best_param()))
        print('best index', index)
        print('feedback dist to center', np.linalg.norm(last_mu - self.es.best_param()))
        # print('forward param mean std: {} max std: {} path length: {}'.format(np.mean(param_std), np.max(param_std), forward_update_dist))


        print(success[0], reward[0], best_raw_reward, worse_raw_reward, self.es.rms_stdev(), self.es.learning_rate, np.linalg.norm(mu_grad), np.linalg.norm(mu_step), from_start, from_last)

        log_metrics = {
                        "ES/mu_reward": reward[0],
                        "ES/best_reward": best_raw_reward,
                        "ES/worse_reward": worse_raw_reward,
                        "ES/reward_std": np.std(reward),
                        "ES/best-mu_reward": best_raw_reward - reward[0],
                        "ES/best-worse_reward": best_raw_reward - worse_raw_reward,
                        "ES/mu_accuracy": success[0],
                        "ES/best_accuracy": success.max(),
                        "ES/worse_accuracy": success.min(),
                        "ES/accuracy_std": np.std(success),
                        "ES/best-mu_accuracy": success.max() - success[0],
                        "ES/best-worse_accuracy": success.max() - success.min(),
                        "ES/sample_better_than_mu": np.sum(np.array(reward)>=reward[0]),
                        "ES/mu_grad": np.linalg.norm(mu_grad),
                        "ES/mu_step": np.linalg.norm(mu_step),
                        "ES/avg_sigma": self.es.rms_stdev(),
                        "ES/sigma_change_max": np.max(sigma_change_ratio),
                        "ES/lr": self.es.learning_rate,
                        "ES/param_dist_from_start": from_start,
                        "ES/param_step": from_last,
                        "ES/best_perfrom_dist to mu": np.linalg.norm(last_mu - self.es.best_param()),
                        "ES/iteration": gen,
                        "Bottom-up/path_len_max": path_len.max(),
                        "Bottom-up/path_len_min": path_len.min(),
                        "Bottom-up/path_len_best": path_len[index],
                    }
        
        # if gen % self.eval_interval == 0:
        #     log_metrics['Bottom-up/train_loss'] = eval_loss
        #     log_metrics['Bottom-up/train_acc'] = eval_acc
        #     log_metrics['Bottom-up/test_loss'] = test_loss
        #     log_metrics['Bottom-up/test_acc'] = test_acc

        return log_metrics


    def save_checkpoint(self, check_path, round, gen):
        with open(check_path+'/{}_{}.check'.format(round, gen), 'wb') as file:
            pickle.dump((self.es.mu, self.es.curr_best_mu, self.es.sigma, self.es.learning_rate, self.es.optimizer), file, protocol=4)
        return '/{}_{}.check'.format(round, gen)

    def restore_checkpoint(self, check_path):

        if os.path.isfile(check_path):
            with open(check_path, 'rb') as handle:
                (self.es.mu, self.es.curr_best_mu, self.es.sigma_init, self.es.learning_rate, self.es.optimizer) = pickle.load(handle)
        else:
            print('There is no {}'.format(check_path))
            assert 0==1 
            
class parallel_simulation():
    def __init__(self, config, population, hid_dim, device, num_worker):
        self.device = device

        self.simulator = simulator(self.device, config=config)

        self.model = Topdown(self.simulator.observation_space.shape[-1], hid_dim ,self.simulator.action_space.shape[-1])
        self.orig_params_flat, self.model_shapes = flat_param(self.model)
        self.NPARAMS = len(self.orig_params_flat)
        self.model.to(self.device)
        self.model.eval()

        self.num_worker = num_worker

        self.trial_count = math.ceil(population / self.num_worker)
        
        self.comm = MPI.COMM_WORLD 
        
        self.size = self.comm.Get_size() 

        assert (self.size-1) == self.num_worker

    def worker(self, gen, rank):
        solution = self.comm.recv(source=0, tag=gen)

        reward = np.zeros((len(solution)))
        param_dist = np.zeros((len(solution)))
        success = np.zeros((len(solution)))
        for i in range(len(solution)):
            update_model(solution[i], self.model, self.model_shapes, self.device)
            episode_reward, info, param_norm = self.simulator.run_one_episode(self.model, data_seed=gen)
            reward[i] = episode_reward
            # episode_R[i] = episode_reward
            param_dist[i] = param_norm
            success[i] = info
        self.comm.send([reward, success, param_dist], dest=0, tag=gen)


    def batch_simulation(self, solutions, gen):

        for i in range(self.num_worker):
            if (i+1)*self.trial_count > len(solutions):
                solution = solutions[i*self.trial_count:]
            else:
                solution = solutions[i*self.trial_count:(i+1)*self.trial_count]
            # print('send',i, time.time())
            self.comm.send(solution, dest=i+1, tag=gen)

        reward_list, param_dist_list, success_list = [], [], []
        for i in range(1, self.num_worker+1):
            result_packet = []
            result_packet = self.comm.recv(source=i, tag=gen)

            reward, success, param_dist = result_packet[0], result_packet[1], result_packet[2]
            reward_list.append(reward)
            param_dist_list.append(param_dist)
            success_list.append(success)

        reward_list = np.concatenate(reward_list, axis=0)
        param_dist_list = np.concatenate(param_dist_list, axis=0)
        success_list = np.concatenate(success_list, axis=0)

        return np.array(reward_list), np.array(success_list), np.array(param_dist_list)


    # def eval(self, mu):
    #     update_model(mu, self.model, self.model_shapes, self.device)
    #     # should train with full dataset before testing
    #     train_loss, train_acc, param_norm = self.simulator.run_one_episode(self.model)
    #     test_loss, test_acc = self.simulator.test_model()
    #     return train_loss, train_acc, test_loss, test_acc