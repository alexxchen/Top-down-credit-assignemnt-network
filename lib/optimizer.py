import numpy as np
import torch

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
  def __init__(self, param, epsilon=1e-08):
    self.param = param
    self.dim = len(param)
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.param
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.param = theta + step
    return self.param, ratio, step

  def _compute_step(self, globalg):
    raise NotImplementedError

class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, param, stepsize, beta1=0.9, beta2=0.999):
    Optimizer.__init__(self, param)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

#   def _compute_step(self, globalg):
#     a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
#     self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
#     self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
#     step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
#     return step

  def _compute_step(self, globalg):
    # Update first moment (momentum-like)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    # Update second moment (adaptive learning rate)
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    
    # Compute bias-corrected estimates
    m_hat = self.m / (1 - self.beta1 ** self.t)
    v_hat = self.v / (1 - self.beta2 ** self.t)
    
    # Calculate step with corrected scaling
    step = -self.stepsize * m_hat / (np.sqrt(v_hat) + self.epsilon)
    return step


def flat_param(model):
    orig_params = []
    model_shapes = []
    for param in model.parameters():
        p = param.data.cpu().numpy()
        model_shapes.append(p.shape)
        orig_params.append(p.flatten())
    orig_params_flat = np.concatenate(orig_params)
    return orig_params_flat, model_shapes

def update_model(flat_param, model, model_shapes, device):
    idx = 0
    i = 0
    for param in model.parameters():
        delta = np.product(model_shapes[i])
        block = flat_param[idx:idx+delta]
        block = np.reshape(block, model_shapes[i])
        i += 1
        idx += delta
        block_data = torch.from_numpy(block).float()
        block_data = block_data.to(device)
        param.data = block_data

