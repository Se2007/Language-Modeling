import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

from benchmark import dataset
import time
import math
import numpy as np
import  wandb
import os
from Methods import LSTM, Transformer
import utils 

key_file = './wandb-key.txt'

if os.path.exists(key_file):
    with open(key_file) as f:
        key = f.readline().strip()
    wandb.login(key=key)
else:
    print("Key file does not exist. Please create the key file with your wandb API key.")

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load(model, loss, optimizer, device='cpu', reset = False, load_path = None):
    model = model
    loss_fn = loss
    optimizer = optimizer

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
            loss_fn.load_state_dict(sate['loss_fun'])
            optimizer.load_state_dict(sate['optimizer'])
            optimizer_to(optimizer, device)
    return model, loss_fn, optimizer
   


def save(save_path, model, optimizer, loss_fn):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_fun' : loss_fn.state_dict()
    }

    torch.save(state, save_path)

def plot(train_hist, valid_hist, label):
    print(f'\nTrained {len(train_hist)} epochs')

    plt.plot(range(len(train_hist)), train_hist, 'k-', label="Train")
    plt.plot(range(len(valid_hist)), valid_hist, 'y-', label="Validation")

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()
    plt.show()

seed = 3
wandb_enable = False

info = {'num_epoch' :90,
        'lr' : 0.1,
        'weight_decay' : 0.00001,
        'device' : 'cuda',
        'reset': False,
        'name_load' : 'TR4.312',#  
        'model_load_path' : './model/Transformer/',
        'model_save_path' : './model/Transformer/'
        }

if wandb_enable:
    wandb_arg_name = input('Please input the WandB argument (run) name:')
    wandb.init(
        project='Language-Modeling',
        name=wandb_arg_name,
        config={
            'lr': info['lr'],
            'weight_decay': info['weight_decay'],
            'num_epoch': info['num_epoch']
        }
    )


if __name__ == '__main__':

    batch_size = 80
    seq_len = 70


    loss_train_hist = []
    loss_valid_hist = []

    metric_train_hist = []
    metric_valid_hist = []

    load_path = info['model_load_path'] + info['name_load'] + ".pth"

    train_loader = dataset.WikiText2('train')(batch_size=batch_size, seq_len=seq_len)
    valid_loader = dataset.WikiText2('valid')(batch_size=batch_size, seq_len=seq_len)
    vocab = dataset.WikiText2('train').vocab


    # model = LSTM.awd_lstm(num_token_vocab=len(vocab))
    model = Transformer.normal_encoder(num_token_vocab=len(vocab))

    optimizer = optim.SGD(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'], momentum=0.9)# , nesterov=True

    loss_fn = nn.CrossEntropyLoss()



    model, loss_fn, _ = load(model, loss_fn, optimizer, device=info['device'], reset = info['reset'], load_path = load_path)

    nonmono = 5
    
    best_val_loss = []
    stored_loss = 100000000
    

    epochs = info['num_epoch']

    for epoch in range(1, epochs+1):
        _, loss_train, metric_train = utils.train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=epoch, device='cuda')
        loss_valid, metric_valid = utils.evaluate(model, valid_loader, loss_fn, device='cuda')

        
        #################################################################
        '''epoch_start_time = time.time()

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2, _ = utils.evaluate(model, valid_loader, loss_fn, device='cuda')
            print(' '+'-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print(' '+'-' * 89)


            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, _ = utils.evaluate(model, valid_loader, loss_fn, device='cuda')
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>nonmono and val_loss > min(best_val_loss[:-nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=info['lr'], t0=0, lambd=0., weight_decay=info['weight_decay'])

            best_val_loss.append(val_loss)'''




        #################################################################

        
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)

        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)


        print(f'Train      - Loss:{loss_train}  Metric:{metric_train}')
        print(f'Validation - Loss:{loss_valid}  Metric:{metric_valid}')
        print()

        if wandb_enable:
            wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                        "metric_valid": metric_valid, "loss_valid": loss_valid})


    save_path = info['model_save_path'] + 'TR' +f'{loss_train:.4}'+ ".pth"
    save(save_path, model, optimizer, loss_fn)

    plot(metric_train_hist, metric_valid_hist, "Metric")
    plot(loss_train_hist, loss_valid_hist, 'Loss')
    