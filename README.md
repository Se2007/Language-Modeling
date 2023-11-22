# Language Modeling Project

## Overview
This project focuses on compare different methods and finding best light weight model for this task.

## Methods
- **LSTM**
  - **Normal LSTM**  A specialized RNN variant with gated cells adept at capturing and utilizing long-term dependencies in sequential data, mitigating vanishing/exploding gradient issues.
  - **AWD-LSTM:** in AWD-LSTM the propose different methods for  regularizing and optimizing LSTMbased models shuch as
- **Transformer Base**
  - **Encoder:** normal encoder block of Transformer.
  - **Decoder:** same as encoder but use Masked Self Attention, This masking ensures that during training, the model doesn't peek ahead to future tokens.
  - **Adaptive Softmax:** circumvents the linear dependency on the vocabulary size by exploiting the unbalanced word distribution to form clusters that explicitly minimize the expectation of computation time. 
   - **Adaptive Input:** Adaptive input consists of dividing the vocabulary into clusters from the most frequent words to the least frequent words that follows a Zipfian distribution, and each cluster has a different dimension that follows word frequency.
  



