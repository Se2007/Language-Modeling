from torch import optim
import dataset
from utils import train_one_epoch
from prettytable import PrettyTable
from torch import nn
from colorama import Fore, Style, init
import torch

import LSTM
import Transformer

def load(model, device='cpu', reset = False, load_path = None):
    model = model

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
    return model

device = 'cuda'

train_loader  = dataset.WikiText2('train', mini=True)(batch_size=80, seq_len=70)
vocab = dataset.WikiText2('train').vocab

reset = False
load_path = './model/Transformer/'+'TR4.475'+ ".pth"

num_epochs = 5

batch_size = 80
seq_len = 70



learning_rates = [1, 0.1, 0.01, 0.001, 0.003, 0.0001]
weight_decays = [1e-3, 1e-4, 1e-5, 5e-5, 1e-6]

loss_list = []

best_lr = None
best_wd = None
best_loss = float('inf')  
min_num = float('inf')
second_min = float('inf')

table = PrettyTable()
table.field_names = ["LR \ WD"] + [f"WD {i}" for i in weight_decays]


for lr in learning_rates:
    for wd in weight_decays:
    
        print(f'\nLR={lr}, WD={wd}')

        loss_fn = nn.CrossEntropyLoss()
        # model = LSTM.awd_lstm(num_token_vocab=len(vocab))
        model = Transformer.normal_encoder(num_token_vocab=len(vocab))
        
        model = load(model, device='cuda', reset = reset, load_path = load_path)
        
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)


        for epoch in range(1, num_epochs+1):
            model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device=device)

     
        loss_list.append(float(f'{loss:.4f}'))

   

sorted_list = sorted(loss_list)
first_min = sorted_list[0]
second_min = sorted_list[1]

first_min_idx = loss_list.index(first_min)
second_min_idx = loss_list.index(second_min)

loss_list[first_min_idx] = f"{Fore.GREEN}{first_min}{Fore.WHITE}"
loss_list[second_min_idx] = f"{Fore.YELLOW}{second_min}{Fore.WHITE}"
loss_list = list(map(str, loss_list))



o = 0

for i in learning_rates:
    row = [f"LR {i}"]

    losses = loss_list[o:len(weight_decays)+o]
    o += len(weight_decays)

    row += losses
    table.add_row(row)


print(table)