import utils
import Transformer
import dataset
import torch
from torch import nn
from torch import optim

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

vocab = dataset.WikiText2('train').vocab
tokenizer = dataset.WikiText2('train').tokenizer

load_path = './model/Transformer/model.pth'

model = Transformer.normal_encoder(num_token_vocab=len(vocab))

model = load(model, device='cuda', reset = False, load_path = load_path)

device = 'cpu'


prompt = 'In a galaxy far, far away, there'
# prompt = ' I am'

indices = vocab(tokenizer(prompt))
itos = vocab.get_itos()

max_seq_len = 35
for i in range(max_seq_len):
  src = torch.LongTensor(indices).to(device).unsqueeze(-1)

  with torch.no_grad():
    prediction = model(src)

  temperature = 0.5
  probs = torch.softmax(prediction[-1]/temperature, dim=0)

  idx = vocab['<ukn>']
  while idx == vocab['<ukn>']:
    idx = torch.multinomial(probs, num_samples=1).item()

  token = itos[idx]
  prompt += ' ' + token

  if idx == vocab['.']:
    break

  indices.append(idx)

print('\n'+prompt)