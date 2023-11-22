from typing import List
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from benchmark.dataset import WikiText2


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
        embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1

  embedding = torch.nn.functional.embedding(words, masked_embed_weight,
                                            padding_idx, embed.max_norm, embed.norm_type,
                                            embed.scale_grad_by_freq, embed.sparse)
  return embedding

def linear_dropout(linear, embbed, dropout=0.1, scale=None):
  if dropout:
    mask = linear.weight.data.new().resize_((linear.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
        linear.weight) / (1 - dropout)
    masked_linear_weight = mask * linear.weight
  else:
    masked_linear_weight = linear.weight
  if scale:
    masked_linear_weight = scale.expand_as(masked_linear_weight) * masked_linear_weight

  return torch.nn.functional.linear(embbed, masked_linear_weight, bias=linear.bias)

  # Linear = torch.nn.functional.embedding(words, masked_embed_weight,
  #                                           padding_idx, embed.max_norm, embed.norm_type,
  #                                           embed.scale_grad_by_freq, embed.sparse)
  # return embedding

###################################################

class AdaptiveInput(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        initial_dim: int,
        factor: float,
        output_dim: int,
        cutoff: List[int],
        emb_dropout = 0.2,
        lin_dropout = 0.2,
    ):
        super().__init__()

        self.emb_dropout = emb_dropout
        self.lin_dropout = lin_dropout


        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()
        self.Linears = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor**i))

            emb = nn.Embedding(size, dim, self.padding_idx)
            lin = nn.Linear(dim, output_dim, bias=False)
            emb.weight.data.uniform_(-0.1, 0.1)
            lin.weight.data.uniform_(-0.1, 0.1)

            # seq = nn.Sequential(
            #     nn.Embedding(size, dim, self.padding_idx),
            #     nn.Linear(dim, output_dim, bias=False)
            # )

            self.embeddings.append(emb)
            self.Linears.append(lin)
            self.padding_idx = None
        self.padding_idx = padding_idx

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        # return self.embeddings[band][0].weight, self.embeddings[band][1].weight
        return self.embeddings[band].weight, self.Linears[band].weight

    def forward(self, input: torch.Tensor):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        # print(result.shape)
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = linear_dropout(self.Linears[i],(embedded_dropout(self.embeddings[i], chunk_input, dropout=self.emb_dropout if self.training else 0)), dropout=self.lin_dropout)
                
            # print(mask)
            # print(chunk_input)

        return result
   
#################################################################

class NormalTransformer(nn.Module):
  def __init__(
      self, num_token_vocab, embedding_dim, nhead_encoder, num_decoder_layer, 
      em_dropout=None
      ):
    super().__init__()

    self.num_token_vocab = num_token_vocab
    self.embedding_dim = embedding_dim
    self.em_dropout  = em_dropout

    self.embedding = nn.Embedding(self.num_token_vocab, self.embedding_dim)
    self.embedding.weight.data.uniform_(-0.1, 0.1)

    decoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead_encoder)
    self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layer)

    self.fc = nn.Linear(embedding_dim, num_token_vocab)

    self.fc.weight = self.embedding.weight


  def forward(self, src):

    if self.em_dropout is not None :
      embedding = embedded_dropout(self.embedding, src, dropout=self.dropoute if self.training else 0)
    else :
      embedding = self.embedding(src)

    encode = self.decoder(embedding)

    predict = self.fc(encode)

    return predict
  
####################################################################  

class AdaptiveInputTransformer(nn.Module):
  def __init__(
      self, num_token_vocab, embedding_dim, nhead_encoder, num_decoder_layer, cutoff, factor=4, 
      emb_dropout=None, encoder_dropout=None
      ):
    super().__init__()

    self.num_token_vocab = num_token_vocab
    self.embedding_dim = embedding_dim
    self.emb_dropout  = emb_dropout
    self.encoder_dropout = encoder_dropout

    self.embedding = AdaptiveInput(
        vocab_size = self.num_token_vocab,
        padding_idx = None,
        initial_dim = self.embedding_dim,
        factor = factor,
        output_dim = self.embedding_dim,
        cutoff = cutoff,
        emb_dropout = self.emb_dropout,
        lin_dropout = self.emb_dropout
    )

    encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead_encoder, dropout=self.encoder_dropout)
    self.decoder = nn.TransformerEncoder(encoder_layer, num_decoder_layer)

    self.fc = nn.Linear(embedding_dim, num_token_vocab)



  def forward(self, src):

    embedding = self.embedding(src)

    encode = self.decoder(embedding)

    predict = self.fc(encode)

    return predict

####################################################################

def normal_encoder(num_token_vocab):
  embedding_dim = 400
  nhead_encoder=8
  num_decoder_layer=6

  vocab = WikiText2('train').vocab

  return NormalTransformer(num_token_vocab=num_token_vocab, embedding_dim=embedding_dim, nhead_encoder=nhead_encoder, num_decoder_layer=num_decoder_layer)

def adptive_input_encoder(num_token_vocab):
  embedding_dim = 512
  nhead_encoder=8
  num_decoder_layer=8
  factor = 4
  cutoff=[150, 1000, 3000]

  emb_dropout = 0.2
  encoder_dropout = 0.4

  return AdaptiveInputTransformer(num_token_vocab, embedding_dim=embedding_dim, nhead_encoder=nhead_encoder, num_decoder_layer=num_decoder_layer,
                                 cutoff=cutoff, factor=factor,emb_dropout=encoder_dropout, encoder_dropout=encoder_dropout)

#######################################################
  

def test(model, device = 'cpu'):
  with torch.no_grad():
    fr = model.to(device)#$$#   torch.randn((10,28),device=device)
    print(fr(torch.LongTensor([[1,2,3,4], [1,2,3,4]])).shape)

def get_parameters(model, device = 'cpu'):
  fr = model.to(device)
  param_values = [param.data for param in fr.parameters()]

  total_params = sum([param.numel() for param in param_values])

  print(f"\nTotal number of parameters: {total_params}")



if __name__ == '__main__':
  embedding_dim = 400
  nhead_encoder=8
  num_decoder_layer=6

  vocab = WikiText2('train').vocab
  # tr = normal_encoder(num_token_vocab=len(vocab))
  # test(tr)
  # get_parameters(tr)

  ai = AdaptiveInput(
        vocab_size = 100,
        padding_idx = None,
        initial_dim = 256,
        factor = 4,
        output_dim = 300,
        cutoff = [30, 70]
    )
  

  print(ai(torch.tensor([1,2,3,40,50,78,80])).shape)

  ait = adptive_input_encoder(num_token_vocab=len(vocab))
  test(ait)
  get_parameters(ait)


