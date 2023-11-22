import utils
import model
import dataset
import torch
from torch import nn
from torch import optim
from train import load


test_loader = dataset.WikiText2(mode='test')(256, 35)
vocab = dataset.WikiText2('train').vocab

load_path = './MImodel_loss0.9142.pth'

model = model.LSTM(len(vocab), 300, 512, 2, 0.65, 0.5)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-05)

model, loss_fn, optimizer = load(model, loss_fn, optimizer, device='cuda', reset = False, load_path = load_path)



loss_test, metric_test = utils.evaluate(model, test_loader, loss_fn, device='cuda')

print(f'Test Loss : {loss_test} -- Perplexity : {metric_test}')
