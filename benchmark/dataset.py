from typing import Any
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2 as WT2
from torchtext.data.utils import get_tokenizer
from torchtext import vocab




def data_process(text_iter, seq_len, tokenizer, vocab):
    data = torch.cat([torch.LongTensor(vocab(tokenizer(line))) for line in text_iter])

    M = len(data) // seq_len

    r = len(data) % seq_len
    # print(r)
    
    data = torch.cat((data, torch.LongTensor([0]))) if r==0 else data

    inputs = data[:M*seq_len]
    targets = data[1:M*seq_len+1]

    inputs = inputs.reshape(-1, seq_len)
    targets = targets.reshape(-1, seq_len)
    
    return inputs, targets

class LanguageModelDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self,):
        return self.inputs.shape[0]
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


    
class WikiText2(Dataset):
    def __init__(self, mode, mini=False):
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini

        train_iter, valid_iter, test_iter  = WT2('./../WikiText2')

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        word_freq_dict = {}
        for tokens in map(self.tokenizer, train_iter):
            for token in tokens:
                if token in word_freq_dict:
                    word_freq_dict[token] += 1
                else:
                    word_freq_dict[token] = 1


        word_freq_pairs = sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True)

        word_to_new_index = {word: index for index, (word, _) in enumerate(word_freq_pairs)}
        self.vocab.stoi = word_to_new_index
            

        if mode == 'train' :
            self.iterable = train_iter
        elif mode == 'valid':
            self.iterable = valid_iter
        elif mode == 'test' :
            self.iterable = test_iter

    def __call__(self, batch_size, seq_len) :
        x, y = data_process(self.iterable, seq_len, self.tokenizer, self.vocab)
        dataset = LanguageModelDataset(x, y)

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        

        elif self.mini == True:
            dataset,_ = random_split(dataset,(5000, len(dataset)-5000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader


if __name__=='__main__':
    loader = WikiText2('test')(1, 115)
    # print(loader.shape)[0].shape
    print(next(iter(loader)))


