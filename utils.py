import torch
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch import nn
import torchmetrics as tm
from torchmetrics import Accuracy






class AverageMeter(object) :
  def __init__(self):
    self.reset()
  def reset(self) :
    self.avg = 0
    self.val = 0
    self.sum = 0
    self.count = 0
  def update (self, val, n=1) :
    self.val = val
    self.count += n
    self.sum += self.val * n
    self.avg = self.sum / self.count



def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None, device='cpu'):
  
  metric = tm.text.Perplexity().to(device)
  model.train().to(device)
  loss_train = AverageMeter()
  metric.reset()

  with tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      inputs = inputs.t().to(device)
      targets = targets.t().to(device)

      outputs =  model(inputs)

      # print(outputs.shape, outputs.reshape(-1, outputs.shape[-1]).shape)
      # print(targets.shape, targets.flatten().unsqueeze(-1).shape)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      loss.backward()

      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=0.25)

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)

      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())



  return model, loss_train.avg, metric.compute().item()

def evaluate(model, test_loader, loss_fn, device='cpu'):
  metric = tm.text.Perplexity().to(device)
  model.eval().to(device)
  loss_eval = AverageMeter()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.t().to(device)
      targets = targets.t().to(device)

      outputs = model(inputs) 

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
      loss_eval.update(loss.item(), n=len(targets))

      metric(outputs, targets)

  return loss_eval.avg, metric.compute().item()




