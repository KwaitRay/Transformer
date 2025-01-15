# Transformer
This project is designed to explore the attention mechanism and how to build an transformer model. The transformer model consists of encoder and decoder block, multi-attention ,self-attention, position-coding is included as well. By collecting the Eng-France dataset online, and through training, I completed a model can be used in text translation.
## Chapter 1. Introduction
### 1.Project Background
- This project is based on the PyTorch framework and aims to build an efficient machine translation model. We used the Transformer model to train on an English-French text dataset to achieve high-quality translation from English to French. In this project, in addition to implementing the basic translation functionality, we also delved into the principles and implementation of the Transformer model, including key modules such as the self-attention mechanism, positional encoding, and the encoder-decoder structure.
- Through this project, you can strengthen your understanding of the deep learning framework Transformer and learned how to integrate theory with practice, completing the entire process from data preprocessing to model training and evaluation. In the future, this project can be further applied to multilingual scenarios or improved through model fine-tuning to enhance translation quality in specific domains.
### 2.Technology Stack
#### Programming Language：Python 3.10.16
#### Deep Learning Framework：PyTorch. torch Version: 2.5.1+cu121
#### Model Architecture：Transformer。
#### Auxiliary Tools：Matplotlib 3.7.2（Visualization）,NumPy 1.24.3（Array Operation）,Pandas 2.0.3（Dataset Reading）
#### Training Tools：GPU（CUDA Support）。
#### Code Management：Git、VS Code。
#### Evaluation and Visualization：Custom Animator Class，show_heatmap Function
## Chapter 2. Transformer Model Architecture and Principles
### 1.Model Architecture
- The Transformer model is a deep learning architecture based on the attention mechanism, divided into two main modules: the encoder and the decoder. The encoder module consists of multiple identical layers, with each layer containing two sub-layers. The first sub-layer is multi-head self-attention aggregation, and the second sub-layer is a position-wise feed-forward network. Specifically, when calculating the self-attention in the encoder, the queries, keys, and values all come from the output of the previous encoder layer. Each sub-layer uses residual connections followed by layer normalization.
- The Transformer decoder is also composed of multiple identical layers stacked together, and residual connections and layer normalization are used in each layer. The decoder layer consists of three sub-layers. The first sub-layer is a masked multi-head self-attention layer, where all inputs depend on the output of the previous decoder layer. Queries, keys, and values all come from the output of the previous decoder layer. In the decoder, each position can only attend to all positions before it. This masking attention retains the auto-regressive property, ensuring that predictions depend only on the previously generated output tokens. The implementation of masked attention is similar to the hidden state implementation in RNNs, where a hidden state is introduced in the forward pass of the module. The second sub-layer is the encoder-decoder attention layer. In encoder-decoder attention, the queries come from the output of the previous decoder layer, while the keys and values come from the entire output of the encoder. The final sub-layer is a position-wise feed-forward network.
  
![transformer_structure](transformer_structure.png)
### 2.Model Principle
#### （1）self-attention Mechanism
  The self-attention mechanism is a key technology in the Transformer model. In terms of implementation, it can be seen as setting the queries, keys, and values in multi-head attention to the same tensor. It allows the model to focus on information from all other elements in the sequence when processing each element of the input sequence. In traditional RNNs, information is dependent on time steps, while self-attention calculates the relevance (i.e., attention weights) between each word, enabling the model to more effectively capture global dependencies. Additionally, position encoding allows the Transformer model to pay attention to the positional information of the input sequence.
  
The computation process of self-attention: Assume there is an input sequence, and we need to calculate the relationship between each word xi and other words in the sequence.
Query、Key、Value：
By performing a linear transformation on the input sequence, three vectors are obtained: Query, Key, and Value. The dimensions of Q, K, and V are usually the same (query_size = key_size = value_size = num_hiddens).

num_hiddens generally represents the number of hidden units, and in the Transformer model, the size of num_hiddens is typically determined based on the features of the input sequence.

Calculating Attention Weights: The attention weights are computed based on the similarity between each query and all keys. The similarity between the query and the key is commonly calculated using dot product：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### （2）Multi-Head Attention
The multi-head attention mechanism calculates self-attention in parallel across multiple "heads." Each head learns different attention patterns in different subspaces, and the outputs of these heads are concatenated together. This allows the model to capture different semantic information from the input sequence, improving model performance. The attention weights for each head are generally computed using dot-product attention.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

![attention_scoring_function](attention_scoring_function.png)
![multi_attention](multi_attention.png)

#### （3）Positional Encoding
Since transformers, unlike RNNs and CNNs, do not have inherent sequential processing capabilities, positional encoding is introduced. Positional encoding is a vector that has the same length as the input sequence and is added to the input embeddings. It helps the model capture the position of words in the sequence. In this project, positional encoding is generated using sine and cosine functions, thereby injecting absolute or relative positions.
## Chapter 3. Technical Details
### （1）Package Import
```python
import random
import torch
from d2l import torch as d2l
import re 
import matplotlib.pyplot as plt
from sequence.text_processing import Vocab 
import pandas as pd
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
```

### （2）Training Data Preprocessing
The training dataset loading and preprocessing is a critical foundation for the model's data. In the internal implementation of d2l.load_data_nmt() in the d2l library, it includes connecting to the d2l data center (DATAHUB) to retrieve the URL corresponding to the 'fra-eng' training document, checking whether the related files exist locally. If not, it sends an HTTP request using the URL through request and responds by downloading the dataset to the local machine. The dataset is then loaded using with open.

Next, the text is preprocessed, which includes operations such as adding spaces before non-empty punctuation marks, converting letters to lowercase, and replacing non-breaking spaces with regular spaces for text normalization. After that, the text sequences are tokenized. Then, using the Vocab module, token-to-index and index-to-token mappings are built, which is a key operation for normalizing the text dataset before feeding it into the model.

Afterward, sequential partitioning can be applied to preserve the contextual dependencies, or random partitioning can be used to enhance the model's generalization ability. Finally, using PyTorch's utils module, torch.utils.data.DataLoader is employed to create iterators. For specific implementation details, you can refer to the corresponding source code in the d2l library. Below is a part of the operations for reading and normalizing the data.

```python
def read_data_nmt():
    """Load the English-French dataset.

    Defined in :numref:`sec_utils`"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """Preprocess the English-French dataset.

    Defined in :numref:`sec_utils`"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
```
#### <1> Tokenize
Tokenization is a key operation in NLP tasks, with the following advantages:
- By breaking down long sequence strings into multiple independent tokens, it improves the processing efficiency for computers.
- By treating words as the basic unit for text analysis, it allows the model to capture the semantics and contextual relationships between words.
- Most models require discrete inputs, and tokenization can adapt to the input format of the model, providing a foundation for subsequent word vector encoding (via the Vocab module).
- It is a critical core module for tasks such as machine translation, sentiment analysis, search engines, and text generation.
#### <2> Vocab
In essence, it is the vocabulary management module, used for vocabulary mapping and index management. It is a key part of NLP technology, transforming tokens into index values that computers can process, with several core functions:
- Maintains a token frequency table, self._token_freq, implemented using the collections.Counter() module from Python's collections library. This token frequency table allows for the counting of tokens, sorting based on frequency, and facilitates the generation of an ordered token index table.
- Initializes the idx_to_token list (which stores tokens in order of index, allowing for easy mapping from index to specific token) and the token_to_idx dictionary (which allows for quick lookup of index corresponding to each token) using the token frequency table (self._token_freq). Through these two lists and dictionaries, the internal transformation between token indices and actual tokens can be automatically handled.
#### <3> Random Partition
Random partition is suitable for scenarios with large datasets and weak context dependencies, such as model pretraining and large-scale distributed training.
#### <4> Sequential Partition
Sequential partition is suitable for scenarios with strong context dependencies and task-specific needs, such as language translation and time series prediction. Both partitioning methods aim to divide the input data and training data into multiple partitions.
#### <5> Iterator Loading
After tokenization and index conversion, the dataset is read line by line. The first element of each line corresponds to the English source word, and the second element is the French target word. These elements are stored in separate arrays and then tensorized. Finally, the source token tensor array (src_array) and the target token tensor (tgt_array) are returned, along with their respective valid lengths (valid_lens). These tensors, along with the valid lengths, are combined into a four-dimensional tensor (src_array, src_valid_lens, tgt_array, tgt_valid_lens). Using the torch.utils.data.DataLoader module, an iterator is created from this tensor. Finally, the iterator (data_iter), source vocabulary (src_vocab), and target vocabulary (tgt_vocab) are returned in the text loading and preprocessing module.
### （3）Multi-Head Attention Mechanism
Through the multi-head attention mechanism, different heads can focus on diverse text features, such as:
- Capturing short-range dependencies in the text sequence, such as the relationship between adjectives and nouns.
- Capturing long-range dependencies in the text sequence, such as the relationship between the subject and predicate when there is a complex modifier.
- Helping the model understand semantic information, where some heads may focus on specific syntactic and semantic patterns.
It also enhances the robustness of the model. When the attention distribution of some heads becomes problematic, other heads can complement or adjust it. The sub-projection matrices in the hidden space also help the model learn complex, high-dimensional representations.
### （4）Self-Attention Mechanism
The self-attention mechanism assigns weights to each element to capture global dependencies between elements:
- Self-attention simultaneously uses the input sequence, after certain transformations, as the query, key, and value for calculating attention weights. During training, it dynamically adjusts the attention weights of each element in relation to itself and other elements in the sequence. In this English-to-French translation project, it captures the dependencies between different components of the sentence.
- It eliminates local position constraints. Unlike RNNs and CNNs, which can only process data sequentially and where the current data can only perceive previous historical data, self-attention treats all elements in the sequence equally. It can simultaneously focus on elements at different positions in the sequence, capturing long-distance dependencies. With the help of position encoding, it does not lose position information.
- The self-attention mechanism adjusts the weights dynamically based on the similarity between each element and other elements in the sequence. Through softmax normalization, it focuses on more relevant information based on the contextual semantics of the input, improving the reliability of decision-making.
### （5）Position Encoding
- Position encoding addresses the issue that the self-attention mechanism cannot capture position information. While allowing self-attention to capture global dependencies, it can also provide either relative or absolute position information. This enables the model to learn the order of elements in the sequence based on position information. Currently, sine and cosine functions are commonly used for position encoding.
- Absolute position encoding provides the specific position of elements, while relative position encoding focuses on the relative relationships between elements.
### （6）Feed-Forward Networks
Feed-forward networks are typically placed after the self-attention layers and can perform nonlinear transformations on each element in the sequence through parallel computation. This enhances the ability to transform local information and capture more complex semantic features, thus obtaining deeper information. It has several advantages:
- Essentially, a feed-forward network consists of two fully connected layers with a nonlinear activation function in between, commonly ReLU. It transforms the input sequence into an output sequence through nonlinear transformations, enabling the model to capture complex patterns in the input data. This allows the model to solve not only simple linear problems but also complex ones.
- Parallel computation greatly increases the model's computational speed, offering higher computational efficiency.
### （7）Layer Normalization
Layer normalization normalizes the data for each layer, improving the stability of the model training process, accelerating convergence, and enhancing model performance:
- Layer normalization normalizes the input for each layer, stabilizing the mean and variance to avoid gradient explosion and vanishing gradients, which improves the stability of the model training process. By normalizing the activation values of each layer, it mitigates the issue of excessively large or small activation values caused by continuous parameter updates.
- Layer normalization makes the input distribution of each layer more stable, leading to smoother parameter updates in the neural network. This reduces fluctuations in parameter adjustments during training, allowing the model to achieve better performance in a shorter time. It also enables setting a larger learning rate, accelerating the convergence speed, without worrying about excessive parameter changes or gradient explosion issues.
- By normalizing the activation values of each layer, layer normalization reduces the differences between training samples, enabling the model to adapt to different inputs and accelerate convergence.
### （8）Encoder Structure
The encoder consists externally of an embedding layer and a position encoding module. Internally, it is made up of num_layers identical layers, each of which includes a self-attention sub-layer and a position-based feedforward network sub-layer. The sub-layers are connected through residual networks and layer normalization.
### （9）Decoder Structure
The decoder consists externally of an embedding layer and a position encoding module. Internally, it is made up of num_layers identical layers. Each layer includes a masked self-attention sub-layer and a feedforward network sub-layer, with an additional encoder-decoder attention sub-layer placed between them. The sub-layers are connected through residual networks and layer normalization. Finally, a fully connected layer nn.Linear(num_hiddens, tgt_vocab) outputs the predicted sequence, mapping from the source sequence to the predicted sequence.
### （10）Encoder-Decoder Coupling
The d2l library provides the d2l.EncoderDecoder() module, which initializes both the Encoder and Decoder modules in the initialization function. During the forward pass, the propagation follows the flow: encoder -> decoder.init_state -> decoder, constructing the transformer model
### （11）Training Process
The sequence-to-sequence training process can be divided into the following steps:
- Xavier initialization of model weights.
- Initialization of model components, including the optimizer (using torch.Adam()) and the loss function (using MaskedSoftmaxLoss).）
- Setting the model to training mode with net.train() and defining an animator (to track the accumulated loss during training and plot the loss curve).
- Iterative training: In each iteration, a timer(d2l.Timer) is used to track the training time, and a metric(d2l.Accumulator(2)) is used to calculate the total loss and token count during each iteration.
- Constructing a tensor consisting of the target sequence (Y) with a batch size of Y.shape[0], filled with <bos> (beginning-of-sequence token), as the initial input for the decoder, and then constructing the decoder input.
- Performing forward propagation and loss calculation (comparing the predicted values from forward propagation with the true labels).
- Calculating the loss, performing backpropagation to ensure the loss is a scalar, clipping the gradients, and updating model parameters. Loss accumulation and token accumulation are done without updating gradients.
- Finally, updating the animator every ten epochs to plot the loss graph.
## Chapter 4. Training Dataset Preprocessing
### （1）Training Dataset Preprocessing
Using the function d2l.load_data_nmt from the d2l library, the internal implementation includes connecting to the d2l data center (DATA_HUB[]), retrieving the URL of the required document. If the file is not found locally, it will be streamed and downloaded to the local machine. Afterward, the file is decompressed, read, and preprocessed, which includes replacing non-breaking spaces (\u02f and \xa0) in the text, converting all letters to lowercase, and adding spaces before punctuation marks that do not have spaces in front. The text is then tokenized, and the tokens are indexed and converted into a form that can be used by the model. Finally, a data iterator is created using torch.utils.data.DataLoader.
```python
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size,num_steps)
```
The following is the key code for internal implementation
#### <1>Tokenize
```python
def tokenize(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines]
    elif token=='char':
        return [list(line) for line in lines]
    else:
        print('Error: unknown token type:'+token)
```
#### <2>Vocab
```python
# Convert token list `tokens` to idx_to_token (initial index-to-token list), and then further convert it to token_to_idx (token-to-index table).
# Use collections.Counter(tokens) to get the _token_freq list, sorted in descending order by frequency.
class Vocab:
    # Special methods like __xxx__ are automatically called when instances of the object are instantiated.
    def __init__(self, tokens=None, min_freq=0, reserved_token=None):
        if tokens is None:
            tokens = []
        if reserved_token is None:
            reserved_token = []
        # Sort tokens based on their frequency (key = lambda x: x[1]), in descending order (reverse=True).
        counter = count_corpus(tokens)
        self._token_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # Initialize idx_to_token list (storing tokens in index order, mapping index back to tokens) and token_to_idx dictionary (using tokens as keys, enabling fast lookup of indices).
        # Special tokens (reserved_token), like <unk> for unknown tokens, <pad> for padding sequences, <bos> for the beginning of sequences, and <eos> for the end of sequences, are added first to idx_to_token.
        self.idx_to_token = ['<unk>'] + reserved_token
        # Initialize token_to_idx using idx_to_token, ensuring that it is an instance attribute (accessible and modifiable), for better encapsulation of the code.
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # Update idx_to_token with the token frequency list and then update token_to_idx.
        for token, freq in self._token_freq:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # For a single token, directly use get to retrieve the corresponding index value. For a list or tuple, recursively call on each token in the list.
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # get is a built-in Python method to retrieve values (i.e., indices) from a dictionary using the key.
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # Convert indices to their corresponding tokens (words), even for lists or tuples, just use self.idx_to_token(index) to retrieve them.
    def __totokens__(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token(index) for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freq(self):
        return self._token_freq

# The function count_corpus() returns a list of token frequency statistics.
def count_corpus(tokens):
    # If the tokens list is empty or if the first element of tokens is a list (i.e., tokens is a 2D list), flatten it.
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # tokens-line-token
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```
#### <3>Random Partition
```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    #"""Generate mini-batches of subsequences using random sampling"""
    # Start partitioning the sequence from a random offset, where the random range includes num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 because we need to account for the labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # Initial indices for subsequences of length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # During the random sampling iteration,
    # subsequences from two adjacent random mini-batches may not be adjacent in the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a subsequence of length num_steps starting from position pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, initial_indices contains random starting indices for subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```
#### <4>Sequential Partition
```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    #"""Generate mini-batches of subsequences using sequential partitioning"""
    # Start partitioning the sequence from a random offset
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```
#### <5>Iterator Loading
```python
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
```
### （2）Attention Function
In order to calculate dot-product attention and additive attention, it's essential to define a masked softmax operation. This operation ensures that elements beyond the valid length are replaced with a very large negative value to avoid influencing the subsequent computations. Additionally, we need to handle the transformation of the valid_lens to ensure it aligns with the input tensor's shape.
```python
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        # If valid_lens is not provided, just apply the standard softmax operation
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # Adjust valid_lens to match the dimensions of X
        if valid_lens.dim() == 1:
            # If valid_lens is 1D, broadcast it to 2D, where the second dimension matches the length of the second dimension of X
            # For example, given valid_lens=[2,3] and shape[1]=2, it repeats the valid_lens as [2,2,3,3] for each sequence
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # If valid_lens is 2D, flatten it into a 1D tensor for easier matching with X's dimensions
            valid_lens = valid_lens.reshape(-1)

    # Flatten X to the shape (batch_size * seq_len, feature_size) to apply the masking
    # Apply the sequence mask to the input tensor X where the masked elements are set to a very large negative value
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

    # Apply the softmax function on the masked tensor and reshape it back to the original shape
    return nn.functional.softmax(X.reshape(shape), dim=-1)
```
####  <1>Dot-Product Attention
This is the foundation of multi-head attention and self-attention, commonly used to compute attention weights.
```python
# Scaled Dot-Product Attention, which is computationally efficient and suitable for GPU computation, widely used in Transformer models.
# DotProductAttention, dropout, queries, keys, values, valid_lens, d, scores, attention_weights
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # valid_lens is set to None by default
    def forward(self, queries, keys, values, valid_lens=None):
        # Scaling using the feature dimension d of queries and keys
        d = queries.shape[-1]
        # In scaled dot-product attention, the feature dimension of queries and keys must be the same, resulting in scores shape (batch_size, query_size, kv_pair_size)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```
#### <2>Additive Attention
```python
class AdditiveAttention(nn.Module):
    # super() is used to query the direct parent class of the current class, and self represents the instance of the current class
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # The shape of queries will be (batch_size, query_size, (1), num_hiddens), keys will be (batch_size, (1), kv_pair_size, num_hiddens)
        # To facilitate broadcasting, we adjust the dimensions, and the final shape of features will be (batch_size, query_size, kv_pair_size, num_hiddens)
        # Note: key_size and value_size are the same, and kv_pair_size refers to the pair size
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # The scores shape will be (batch_size, query_size, kv_pair_size) after removing the last dimension num_hiddens
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # The shape of values is (batch_size, kv_pair_size, value_size), and after applying the attention weights, the output shape will be (batch_size, query_size, value_size)
        # Each row of the output matrix can be viewed as the context representation of a query
        return torch.bmm(self.dropout(self.attention_weights), values)
```
### （3）Multihead Attention
```python
# Multi-Head Attention Model
# MultiHeadAttention, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias, attention, W_q, W_k, W_v, W_o
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# To reshape the matrices for multiple attention heads
# From input X(batch_size, kv_pair_size, num_hiddens) -> (batch_size, kv_pair_size, num_heads, num_hiddens/num_heads)
# -> (batch_size, num_heads, kv_pair_size, num_hiddens/num_heads) -> (batch_size*num_heads, kv_pair_size, num_hiddens/num_heads)
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# Restore to the original shape
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```
### （4）self-attention
```python
# The self-attention mechanism essentially uses the input data as query, key, and value, 
# and the multi-head attention model is used to find correlations between the input data.
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()  # Set the model to evaluation mode

# batch_size, num_queries, num_kv_pairs, valid_lens, X, Y
batch_size, num_queries = 2, 4
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))

# Get the shape of the output when passing X as the query, key, and value
output_shape = attention(X, X, X, valid_lens).shape
```
### （5）Position Encoding 
```python
# Positional Encoding, using sine and cosine functions for fixed position encoding. 
# Sine function is used for even dimensions and cosine function for odd dimensions.
# PositionalEncoding class, with attributes such as num_hiddens, dropout, max_len, P, X, dtype.
class PositionalEncoding(nn.Module):
    # No need to introduce **kwargs as the calculation method and the function are fixed,
    # and there's no need for dynamic or uncertain parameters.
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # torch.zeros() is used to create a matrix of zeros with the given shape.
        self.P = torch.zeros((1, max_len, num_hiddens))
        
        # The positional encoding maps the time step or sequence index (max_len) 
        # to feature dimensions (num_hiddens). Using sine and cosine functions of 
        # different frequencies, we provide the model with positional awareness.
        # The following is the formula implementation.
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        # self.P's shape is (1, num_steps, num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # Even dimensions use sine
        self.P[:, :, 1::2] = torch.cos(X)  # Odd dimensions use cosine
    
    def forward(self, X):
        # Align the second dimension of P with the second dimension of X (num_steps) 
        # using X.shape[1], the other dimensions are automatically broadcast.
        # Then, place P on the same device as X (GPU/CPU), otherwise computation will fail.
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```
### （6）Feedforward Neural Network
```python
# Constructing a position-wise feedforward network, which independently maps each input feature without regard to its position. 
# It consists of two fully connected layers and an activation layer.
# PositionWiseFFN, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, dense1, relu, dense2
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```
### （7）Layer Normalization
```python
# Pass the data processed by the feedforward network (with random dropout applied) along with the input data, 
# and perform a residual connection followed by layer normalization.
# AddNorm, normalized_shape, dropout
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    # In the forward pass of AddNorm, input X from the previous layer and output Y from the current layer are passed.
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```
### （8）Encoder Structure
```python
# Build the Transformer encoder block, with attention -> addnorm1 -> ffn -> addnorm2. 
# use_bias controls whether to use bias terms in the layer computations.
# EncoderBlock, key_size, query_size, value_size, num_hiddens, norm_shape, 
# ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias
# attention, addnorm1, ffn, addnorm2, X, valid_lens, Y
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, 
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# Build the Transformer encoder, which includes embedding layer -> positional encoding -> num_layers EncoderBlock modules.
# Compared to the encoder block, it also initializes vocab_size to map the vocabulary to the hidden layer (feature dimension),
# and num_layers to stack multiple Transformer blocks.
# TransformerEncoder, vocab_size, key_size, query_size, value_size, num_hiddens, 
# norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias
# num_hiddens, embedding, pos_encoding, blks, 'block'
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                              norm_shape, ffn_num_input, ffn_num_hiddens, 
                                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        # Since the blocks container stores the (index 'block x' and corresponding model pairs),
        # we use enumerate to iterate and obtain both the index and the model.
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # This attention weight path is: blks -> blk -> d2l.MultiAttention -> d2l.DotAttention -> corresponding self.attention_weights
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
```
### （9）Decoder Structure
```python
# Construct the decoder module, note the training and inference stages
# DecoderBlock: key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs
# attention1, addnorm1, attention2, addnorm2, ffn, addnorm3
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    # X, state, enc_outputs, enc_valid_lens, key_values, training, batch_size, num_steps, dec_valid_lens, X2, Y, Y2, Z
    def forward(self, X, state):  # Returns the normalized output along with the encoder's output, valid lengths, and the current and historical inputs of the decoder
        # Use the saved current decoder input and historical outputs in the state to obtain the encoder outputs (as decoder inputs) and the valid lengths of the encoded sequence
        enc_outputs, enc_valid_lens = state[0], state[1]
        # state[2] stores the historical inputs of the current decoder block. If there's no history, use the current input as key_values, then pass it back to state[2][self.i]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        # Note that key_values include both the current decoder input X and the historical data, so in multi-head attention, it can be viewed as self-attention
        state[2][self.i] = key_values
        # Depending on whether it's training, adjust the decoder's valid lengths based on the shape of input X (batch_size, num_steps, num_hiddens)
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Multi-head attention: use decoder output (Y) and encoder input (enc_outputs)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# Build the decoder structure: embedding layer -> positional encoding -> num_layers decoder_blk(blks) -> fully connected layer
# TransformerDecoder, AttentionDecoder, vocab_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_outputs, ffn_num_hiddens
# num_heads, num_layers, dropout, **kwargs, embedding, pos_encoding, blks
class TransformerDecoder(d2l.AttentionDecoder):
    # Bias term is not needed
    def __init__(self, vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                                            norm_shape, ffn_num_input, ffn_num_hiddens, 
                                                            num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # enc_outputs, enc_valid_lens, *args
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    # X, attention_weights, blks, i, blk
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # Initialize the attention weight container, which needs two groups: one for storing self-attention weights and the other for encoder-decoder attention weights
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        # Since the blks container stores (index 'block x' and its corresponding model key-value pair), we use enumerate to traverse them simultaneously
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # The attention weight path is: blks -> blk -> d2l.MultiAttention -> d2l.DotAttention -> corresponding self.attention_weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```
### （10）Model Parameter Initialization and Encoder-Decoder Coupling
```python
#Model Parameter Initialization
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10  
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size,num_steps)
encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)

```
### （11）Training Process
To perform sequence-to-sequence training, you can use the pre-packaged training module d2l.train_seq2seq()
```python
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```
## Chapter 5. Experiments and Results Analysis
### 1.Detection Experiment Design
#### (1)Design Objective
In this experiment, the 'fra-eng' text dataset from the d2l Data Center DATAHUB is used as the training dataset. After preprocessing, it is loaded into the designed transformer model for training. By performing parameter fine-tuning, observing the loss function curve, and testing the results, we aim to determine the hyperparameters and ultimately achieve reliable translation performance. The purpose of this translation model design is to build an efficient and accurate translation tool to facilitate cross-lingual communication, particularly for English-French translation. By combining the currently popular transformer model, it captures the multi-level, complex semantic features between source languages and generates fluent and natural translations in the target language. This model meets the user's translation needs across various scenarios.
#### (2)Test Code
```python
# Test the trained model by selecting English-French sentence pairs and predicting translations
# If the input string contains a single quote, it should be escaped with \ to indicate it is part of the string, not the end of the string
engs = ['go .', 'i lost .', 'he\'s calm .', 'I\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    # Use d2l.predict_seq2seq to get the translation and decoder attention weight sequence, then print the translation and BLEU score for evaluation.
    # k=2 indicates the accuracy of bigram combinations.
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ', f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

# Get the encoder self-attention weights and visualize them
# enc_attention_weights
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
d2l.show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
                  figsize=(7, 3.5))

# Get both the decoder self-attention weights and the encoder-decoder attention weights
# dec_attention_weights_2d, step, dec_attention_weight_seq, attn, blk, head, dec_attention_weights_filled, dec_attention_weights
# dec_self_attention_weights, dec_inter_attention_weights
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weight_seq 
                            for attn in step 
                            for blk in attn 
                            for head in blk]
dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
# Unpack and assign values, extracting the two parts along the 0th dimension: dec_self_attention_weights: self-attention weights. dec_inter_attention_weights: interaction attention weights
dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
# Plot the heatmaps
d2l.show_heatmaps(dec_self_attention_weights, xlabel='Key positions', ylabel='Query positions', 
                  titles=['head %d' % i for i in range(1, 5)], figsize=(7.5, 3))
d2l.show_heatmaps(dec_inter_attention_weights, xlabel='Key positions', ylabel='Query positions', 
                  titles=['head %d' % i for i in range(1, 5)], figsize=(7.5, 3))
plot.show()
```
### 2.Results Analysis
#### (1).Loss Curve
The loss function reflects the gap between the model's predicted values and the actual values. When the loss curve decreases too slowly or oscillates, it may indicate that the learning rate is either too high or too low. If the loss value suddenly increases or stagnates, it may indicate issues such as gradient explosion or vanishing gradients. In the model constructed here, the ReLU activation function was chosen for the feedforward network to avoid vanishing gradients. During training, Xavier initialization and gradient clipping were used to limit the maximum gradient norm, thereby preventing gradient explosion. Since the transformer model has num_layers core layers in both the encoder and decoder, residual connections (where the output of the current layer y(x) is added directly to the input of the previous layer x, seen in the Addnorm module for residual connection and layer normalization) are used between sub-layers of each layer to avoid gradient explosion.
- The animator module in the training function d2l.train_seq2seq() was used to track the change in the loss function during training in real time, and updates were made based on this change, with images being plotted.
- After adjusting the hyperparameters, it was found that when the learning rate was set to 0.005 and the training batch size was 200, the loss curve for the transformer model on the 'fra-eng' training dataset (obtained from d2l.load_data_nmt(batch_size, num_steps)) showed a steady decline with each training epoch. The speed of the decline gradually slowed down: it dropped the most quickly between epochs 0-50, continued to decrease between epochs 50-100 with some fluctuations, and then stabilized, ultimately converging around 0.025, showing good training results.
![loss_function_curve](loss_function_curve.png)
#### (2).Attention Weight Heatmaps
- Attention weights represent the correlation between elements at different index positions in the key and query sequences. Since the transformer model is based on the self-attention mechanism, the attention weight heatmaps in the encoder and decoder reflect the correlation between elements within the sequence. If a sequence pays more attention to itself, the values on the diagonal will be darker. Higher weights between non-adjacent elements indicate strong semantic and contextual relationships between positions, such as between the subject and verb in a sentence, even if they are far apart and separated by modifiers. By observing the attention weight heatmaps, we can also analyze whether the model correctly understands the global context and whether there is local attention or missing context.
- Different attention weights for different heads provide various semantic information. By examining the heatmaps, we can observe the relationships between different parts of the sequence. By applying the attention weights to the values and aggregating them through multi-head attention, the model eventually obtains a comprehensive representation of contextual features.
- Encoder self-attention weights.

![encoder_attention_weights](encoder_attention_weights.png)

- Dncoder self-attention weights.
  
![decoder_self_attention_weights](decoder_self_attention_weights.png)

- Encoder-Decoder Attention Weights:
  
![decoder_self_attention_weights](decoder_inter_attention_weights.png)
#### 3.Test Results Display (BLEU)
Load some English sentences for model testing and display the prediction results. The evaluation is performed using the commonly used automatic metric in machine translation, BLEU (Bilingual Evaluation Understudy). This metric:
- Calculates the n-gram match between the machine-generated translation and one or more reference translations.
- Measures translation quality by taking the weighted average of n-gram matches across multiple lengths.
- Introduces a penalty mechanism (Brevity Penalty, BP) to prevent the model from obtaining a high score by simply copying short segments of the reference text.
  The formula for BLEU is as follows:
  
#### The BLEU (Bilingual Evaluation Understudy) score calculation formula

1. **The overall BLEU score formula**：
   
$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \cdot \log p_n\right)
$$

: Brevity Penalty

N: The highest order of n-grams (usually 4)

w_n: The weight for the nth n-gram precision，(usually 1/N)

p_n : The precision of the nth n-gram:

  
$$
p_n = \frac{\sum_{c \in C'} \text{count}(c)}{\sum_{c \in C} \text{count}(c)}
$$

C ′is the set of n-grams in the reference translations.
𝐶C is the set of n-grams in the candidate translation.
count(𝑐)count(c) is the count of each n-gram 𝑐c in the respective set.
2. **The length penalty factor（BP）**：

$$
\text{BP} =
\begin{cases} 
1 & \text{if } c > r \\
\exp(1 - \frac{r}{c}) & \text{if } c \leq r
\end{cases}
$$

c: candidate length

r: reference length


loss 0.029, 5164.5 tokens/sec on cuda:0
| original sentence => target sentence       | bleu       | 
|------------|------------|
|go .=>va ! |  bleu 1.000|
|i lost .=>j'ai perdu .|  bleu 1.000|
|he's calm .=>il est calme .|  bleu 1.000|
|I'm home .=>je suis chez moi .|  bleu 1.000|

## Chapter 6. Technical Challenges and Personal Reflections
### 1.Tokenization and Token Index Mapping
Tokenization is an essential step in NLP. It involves dividing long string sequences into individual tokens, which are then converted into token indices. This is crucial for building the vocabulary table. During token index conversion, the main challenge lies in maintaining a frequency table and constructing a vocab module based on it. Inside this module, the frequency table elements are sorted by their frequency. The updated frequency table is then used to fill self.idx_to_token, which in turn updates self.token_to_idx. This design is efficient and ensures that the token indices map correctly to the corresponding tokens in the vocabulary. By leveraging the frequency table, the approach simplifies and streamlines the code, making it efficient.
### 2.Tensor Data Format Conversion During Training
Normalizing the input data format during training is a key factor in ensuring the model operates smoothly. The process can be broken down as follows:
- Initially, the text is loaded as a single string.
- Through tokenization, the string is converted into a sequence of token indices, which forms a 1D sequence of length equal to len(self.idx_to_token).
- After sequential partitioning, the dataset is structured into batches of size (batch_size, num_steps), where each element belongs to the vocabulary, i.e., the values are mapped to indices in src_vocab and tgt_vocab respectively.
- In the model, these sequences are mapped to the hidden layer, with the hidden layer having num_hiddens units. The process is as follows: vocab_size -> embedding layer -> hidden layer. The hidden layer extracts features from the embeddings, converting the low-dimensional discrete features from the vocabulary into continuous features that are suitable for the model. This helps the model learn complex relationships between different languages.
- This structured data is then fed into the transformer model as a (batch_size, num_steps, num_hiddens) tensor for further processing.
### 3.Parameter Matching
n the transformer model, parameters such as query_size, key_size, value_size, and num_hiddens are often set to the same value in the industry. In many cases, num_hiddens is used as a substitute for the other parameters to simplify the model configuration.
## Chapter7. Reference Code
This section includes the specific implementations of various modules and test cases.
- [attention_cues](attention_cues.py) Implements the functionality for drawing attention heatmaps and the Nadaraya-Watson kernel regression.
- [attention_scoring_function](attention_scoring_function.py) Implements attention weight functions, including additive attention and dot-product attention.
- [bahdanau_attention](bahdanau_attention.py) A simple sequence-to-sequence model designed with an encoder-decoder architecture based on two recurrent neural networks.
- [multi_and_self_attention](multi_and_self_attention.py) Implements multi-head attention mechanism and self-attention mechanism.
- [transformer](transformer.py) The implementation of the transformer model.
