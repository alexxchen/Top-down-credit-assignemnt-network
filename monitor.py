import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import shutil
import time
import torch
from mpi4py import MPI
from evolve_mpi import Evolution, parallel_simulation
import wandb
from lib.optimizer import Adam, BasicSGD, SGD

generation = 2000
global_device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset='MNIST'
population = 801
if global_device == 'cpu':
    worker = 201
    worker_per_device = 4
else:
    worker = 56
    worker_per_device = 7
hidden_width = 30
pgpe_config = {                    
                "optimizer": SGD,
                "popsize": population,
                "sigma_init": 0.1,
                "sigma_decay": 0.999,
                "sigma_alpha": 0.1,
                "sigma_limit": 0.0001,
                "sigma_max_change": 0.2,
                "learning_rate": 0.1,
                "learning_rate_decay": 1.0, # annealing the learning rate
                "learning_rate_limit": 0.001,   # stop annealing learning rate
                "average_baseline": False,
                "rank_fitness": True,
                "forget_best": True,
                "weight_decay": 0.0,
                "top_down_hidden_width": hidden_width
                }

class_idx = [0,1,2,3,4,5,6,7,8,9]
n_way = 10

forward_config = {
    "dataset": dataset,
    "class_labels": class_idx,
    "n_way": n_way, 
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
}

use_wandb = True
wandb_project_name = 'Top-down-mnist'
wandb_entity = None
group = None
notes = None
run_name = '{}-{}-{}_class'.format(dataset, forward_config["samples_per_class"], forward_config["n_way"])

restore = False
restore_from = 'top-down-network.check'

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        os.mkdir(os.path.join(path, 'check'))
        os.mkdir(os.path.join(path, 'log'))
        
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class Monitor():
    def __init__(self, device):
        if use_wandb:
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                group=group,
                notes=notes,
                name=run_name,
                config={**pgpe_config, **forward_config},
                save_code=True,
            )
            wandb.run.log_code(".")

        self.generation = generation
        self.evolution = Evolution(forward_config, pgpe_config, population, hidden_width, worker, self.generation, device)

        self.device = device

        self.path = 'Experiment-history'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_path = self.path + '/' + time.strftime("%Y%m%d-%H%M%S")

        create_exp_dir(self.save_path, scripts_to_save=glob.glob('*.py'))

        if restore:
            self.evolution.restore_checkpoint(restore_from)

    def one_run(self, round):
        for gen in range(self.generation):
            print("Generation:", gen)
            log_metrics = self.evolution.evolve_model(gen)
            log_metrics["ES/iteration"] = gen
            if gen % 1 == 0:
                self.evolution.save_checkpoint(os.path.join(self.save_path, 'check'), round, gen)
                if use_wandb:
                    wandb.log(log_metrics, step=population * self.evolution.parallel_simulator.simulator.episode * gen)
        file_name = self.evolution.save_checkpoint(os.path.join(self.save_path, 'check'), round, gen)
        return file_name

    def reset_evolution(self):
        self.evolution = Evolution(pgpe_config, population, hidden_width, worker, self.generation, self.device)

    def train_model(self):
        for i in range(1):
            print('[Round %d]'%i)
            file_name = self.one_run(i)
            # self.reset_evolution()
        if use_wandb:
            wandb.save(os.path.join(self.save_path, 'check')+file_name)
            


if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank() 

    if 'cuda' == global_device:
        print(torch.cuda.device_count())
        assert torch.cuda.device_count() * worker_per_device >= worker

    if rank == 0:
        print('main process started')
        print('thread num: OMP',os.environ['OMP_NUM_THREADS'], 'MKL', os.environ['OMP_NUM_THREADS'])
        
        monitor = Monitor('cpu')
        monitor.train_model()
    else:
        if 'cuda' == global_device:
            device = (rank - 1) // worker_per_device
            print('start worker {} on cuda:{}'.format(rank, device))
            simulation = parallel_simulation(forward_config, population, hidden_width, 'cuda:{}'.format(device), worker)
        else:
            print('start worker {} on cpu'.format(rank))
            simulation = parallel_simulation(forward_config, population, hidden_width, 'cpu', worker)

        for gen in range(generation):
            simulation.worker(gen, rank)