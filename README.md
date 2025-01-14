# Transformer
This project is designed to explore the attention mechanism and how to build an transformer model. The transformer model consists of encoder and decoder block, multi-attention ,self-attention, position-coding is included as well. By collecting the Eng-France dataset online, and through training, I completed a model can be used in text translation.
关键在项目问题以及解决方案，个人经历和贡献，理解和思考
## 一.介绍
### 1.项目背景
- 本项目基于 PyTorch 框架，旨在构建一个高效的机器翻译模型。我们使用了 Transformer 模型对英语-法语文本数据集进行了训练，以实现从英文到法语的高质量翻译功能。在项目中，除了实现基础翻译功能外，还深入学习了 Transformer 模型的原理与实现，包括自注意力机制、位置编码、编码器-解码器结构等关键模块。
- 虽然当前项目主要聚焦于学习与实现，但我们也探索了部分模型优化策略，为未来的多语言翻译扩展奠定了基础。通过本项目，我不仅强化了对深度学习框架的理解，还掌握了如何将理论与实践结合，完成从数据预处理到模型训练与评估的全流程。未来，该项目可以进一步应用于多语言场景，或通过模型微调提升特定领域翻译质量。
### 2.技术栈
#### 编程语言：Python 3.10.16
#### 深度学习框架：PyTorch. torch Version: 2.5.1+cu121
#### 模型架构：Transformer。
#### 辅助工具：Matplotlib 3.7.2（可视化）,NumPy 1.24.3（数组运算）,Pandas 2.0.3（数据集读取）
#### 训练工具：GPU（CUDA 支持）。
#### 代码管理：Git、VS Code。
#### 评估与可视化：自定义的 Animator 类，show_heatmaps函数
详细记录背景和假设：确保对项目的背景、目标和假设做详细清晰的说明，帮助他人理解你的工作。
清晰描述方法和过程：对于方法论和实验步骤，要描述清晰，确保别人能够根据你的描述理解每一个关键的步骤，包括算法原理、数据处理流程、模型设计等。
提供充分的实验验证：如果项目包含实验，确保数据集、实验设置、评估指标等都描述明确，并且提供必要的结果支持，避免模糊不清的结果或解释。
使用清晰的语言和格式：避免过于复杂的语言和专业术语，尽量使用简洁明了的表达，使得描述可以被广泛理解。适当时可以配合图表、公式等辅助说明。
## 二.Transformer模型结构和原理
### 1.模型结构
- Transformer模型是基于注意力机制的深度学习模型，模型分成编码器和解码器两大模块，编码器模块由多个相同的层来组成，每个层都有两个子层。第一个子层是多头自注意力汇聚；第二个子层是基于位置的前馈网络。具体来说，在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出。每个子层都采用了残差连接加法计算，然后再应用层规范化。

- Transformer解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和层规范化。解码器层由三个子层构成，第一个子层是掩蔽多头自注意力层，该层的输入全都依赖于解码器的上一层，查询，键，值都来自上一个解码器层的输出。解码器中的每个位置只能考虑该位置之前的所有位置。这种掩蔽（masked）注意力保留了自回归（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。掩蔽注意力的实现与RNN中的隐状态实现有相似之处，都在模块的前向传播函数中引入了隐状态state。第二个层是编码器－解码器注意力（encoder-decoder attention）层。在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出。最后一个子层是一个基于位置的逐位前馈网络。
  
![transformer_structure](transformer_structure.png)
### 2.模型原理
#### （1）自注意力机制(self-attention)
  自注意力机制是Transformer的关键技术，实现上可以看作是将多头注意力中的查询，键，值都设置成相同张量，它允许模型在处理一个输入序列的每个元素时，能够关注到该序列中所有其他元素的信息。在传统的RNN中，信息是按时间步骤依赖的，而自注意力机制则通过计算每个词之间的相关性（即注意力权重），让模型更有效地捕获全局依赖。同时通过位置编码，使得transformer模型能够关注到输入序列的位置信息
  
自注意力的计算过程：
假设有一个输入序列，我们需要为每个词xi​计算与序列中其他词的关系：
查询（Query）、键（Key）、值（Value）：
通过对输入序列进行线性变换得到三个向量：查询Query，键Key，值Value. Q,K,V 的维度通常相同
(query_size = key_size = query_size = num_hiddens)

num_hiddens一般表示隐藏层单元数量，在transformer中num_hiddens大小一般根据输入序列特征数确定

计算注意力权重：
计算的是每个查询与所有键的相似度。使用查询和键计算相似度，常用的是点积计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### （2）多头注意力（Multi-Head Attention）
多头注意力机制通过多个“头”并行计算自注意力，每个头在不同的子空间上学习不同的注意力模式，最终将这些头的输出拼接起来。从而可以学习输入序列不同的语义信息，提高模型表现。每个头计算注意力权重一般也是采用点积注意力

#### （3）位置编码（Positional Encoding）
由于transformer不像RNN,CNN具有顺序处理能力，因此引入了位置编码，位置编码是一个与输入序列长度相同的向量，它加入到输入嵌入（embedding）中，可以帮助模型捕捉词语在序列中的位置。本项目中通过正弦余弦函数生成位置编码，从而注入绝对或是相对位置
## 三.技术细节
### （）包导入
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
描述模块的设计动机、理论支撑和高层逻辑。
示例：多头注意力机制为什么能够提升模型效果？
### （）训练数据集预处理
#### <>文本加载
#### <>分词
#### <>vocab模块
#### <>随机分区
#### <>顺序分区
### （）多头注意力机制
### （）自注意力机制
### （）位置编码
### （）前馈网络
### （）层规范化
### （）编码器结构
### （）解码器结构
### （）编码器解码器耦合
### （）训练流程
## 四.关键代码段实现
展示具体实现方式和工具的使用。
示例：torch.nn.MultiheadAttention 的实现如何调用？
### （）训练数据集预处理
使用d2l库中封装好的函数d2l.load_data_nmt，内部实现主要包括链接到d2l数据中心DATA_HUB[]，返回所需文档的url,如果查询本地不存在该文件，利用流式传输下载到本地，然后进行解压，读取，预处理，包括替换文本中的非破坏性空格（'\u02f'和'\xa0'）,将所有字母替换为小写以及在前面没有空格的标点符号前插入空格，然后对文本进行分词，词元索引转化，利用torch.utils.data.DataLoade构建数据迭代器
```python
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size,num_steps)
```
以下是内部实现的关键代码
#### <>分词tokenize
```python
def tokenize(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines]
    elif token=='char':
        return [list(line) for line in lines]
    else:
        print('错误，未知词元类型:'+token)
```
#### <>vocab模块
```python
#将词元列表tokens转变为idx_to_token(初始索引-词元列表)，然后进一步转变为token_to_idx(词元-索引表)
#利用collections.Counter(tokens)来得到_token_freqs列表，按照词频进行排序，降序排列
class Vocab:
    #__xxx__特殊方法在使用实例化对象时会自动调用
    def __init__(self,tokens=None,min_freq=0,reserved_token=None):
        if tokens is None:
            tokens=[]
        if reserved_token is None:
            reserved_token=[]
        #按照token的出现频率(key = lambda x:x[1])进行排序,降序排列(reverse=True)
        counter = count_corpus(tokens)
        self._token_freq=sorted(counter.items(),key=lambda x : x[1] ,reverse=True)
        #然后对idx_to_token列表（按索引顺序存储词元，以便将索引映射回具体词元）以及token_to_idx字典（按词元即键值，快速查找对应的索引)
        #以此进行初始化，优先在idx_to_token中加入特殊token(reserved_token),比如<unk>表示词表中的未知词元，<pad>对序列进行填充，<bos><eos>序列的起始和结束标记
        self.idx_to_token = ['<unk>']+reserved_token
        #用idx_to_token来初始化token_to_idx,注意要用self.，表示是Vocab的实例的属性，便于之后进行访问与修改，保证了代码的封装性
        self.token_to_idx = {token:idx for idx,token in enumerate(self.idx_to_token)}
        #利用词频表更新idx_to_token,从而更新token_to_idx
        for token,freq in self._token_freq:
            if freq<min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
    def __len__(self):
        return len(self.idx_to_token)
    #单个词元就直接用get获取该token对应的索引值，列表或者元组就对tokens中的每个token进行递归调用
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            #get是python内置的通过字典的键来获取值(即索引)的方法
            return self.token_to_idx.get(tokens,self.unk )
        return [self.__getitem__(token) for token in tokens]
    #将索引(indices)转换成对应的词元(token),为列表或者tuple也只要用self.idx_to_token(index)直接获取
    def __totokens__(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token(index)for index in indices]
    @property
    def unk(self):
        return 0
    @property
    def token_freq(self):
        return self._token_freq
```
#### <>随机分区
```
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    #"""使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```
#### <>顺序分区
```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    #"""使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
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
#### <>迭代器加载
```python
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
```
### （）注意力函数
计算点积注意力以及加性注意力需要提前定义好掩蔽softmax操作，用于将超过有效长度的元素用一个非常大的负值代替，避免影响后续计算，同时处理好有效长度valid_lens的转化
```python
def masked_softmax(X,valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        shape = X.shape
        #进行valid_lens维度变换，本质上是为了与输入张量X进行维度匹配
        if valid_lens.dim() == 1:
            #当valid_lens是一维时，将其广播为二维，第二维的长度与X的第二维相匹配,比如[2,3]，假设shape[1]=2，每个样本有两个序列，对应每个序列[2,2,3,3]
            valid_lens = torch.repeat_interleave(valid_lens,shape[1])
        else:
            #当valid_lens是二维时，为了避免维度不匹配，将其展开为一维，方便与输入张量X的维度相匹配
            valid_lens = valid_lens.reshape(-1)
    #X展平为(batch_size * seq_len, feature_size)形状,可以堪称批次数量（样本数）,序列数,每个序列的特征值数量
    X = d2l.sequence_mask(X.reshape(-1,shape[-1]),valid_lens,value=-1e6)
    return nn.functional.softmax(X.reshape(shape),dim=-1)
```
####  <>点积注意力
这是多头注意力和自注意力的基础，常用于计算注意力权重
```python
#缩放点积注意力,计算效率高，适合在GPU上进行计算，广泛应用于transformer模型中
#DotProductAttention,dropout,queries,keys,values,valid_lens,d,scores,attention_weights
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention,self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
    #valid_lens初始值设置为None
    def forward(self,queries,keys,values,valid_lens=None):
        #利用queries以及keys的特征数d(shape[-1])进行缩放
        d = queries.shape[-1]
        #在缩放点积注意力中，要求queries和keys的特征数相同,对应scores形状为(batch_size,query_size,kv_pair_size)
        scores =torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)
```
#### <>加性注意力
```python
class AddictiveAttention(nn.Module):
    #super()传入当前类名，表示向上查询当前类的直接父类，self表示传递当前类的直接实例
    def __init__(self, key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AddictiveAttention,self).__init__(**kwargs)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens):
        #对应的queries最后形状为(batch_size,query_size,(1),num_hiddens),keys形状为(batch_size,(1),kv_pair_size,num_hiddens),在features计算中
        #为了便于广播，需要调整维度,最后得到的features形状为(batch_size,query_size,kv_pair_size,num_hiddens),注意key_size与value_size相同，pairsize
        queries,keys = self.W_q(queries),self.W_k(keys)
        features = queries.unsqueeze(2)+keys.unsqueeze(1)
        features = torch.tanh(features)
        #scores删除了最后一维num_hiddens,因此形状为(batch_size,query_size,kv_pair_size)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        #values形状为(batch_size,kv_pair_size,value_size),因此应用权重矩阵后为(batch_size,query_size,value_size)
        #输出矩阵的每一行可以视为一个查询的上下文表示
        return torch.bmm(self.dropout(self.attention_weights),values)
```
### （）多头注意力
```python
#进行多头注意力模型建模
#MultiHeadAttention,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias,attention,W_q,W_k,W_v,W,_o
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias = False,  **kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size,num_hiddens,bias = bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias = bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias = bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias = bias)
    def forward(self,queries,keys,values,valid_lens):
        queries = transpose_qkv(self.W_q(queries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)
        output = self.attention(queries,keys,values,valid_lens)
        output_concat = transpose_output(output,self.num_heads)
        return self.W_o(output_concat)

#为了多注意力的头进行矩阵形状转换
#从输入X(batch_size,kv_pair_size,num_hiddens)->(batch_size,kv_pair_size,num_heads,num_hiddens/num_heads)
#->(batch_size,num_heads,kv_pair_size,num_hiddens/num_heads)->(batch_size*num_heads,kv_pair_size,num_hiddens/num_heads)
def transpose_qkv(X,num_heads):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])
#恢复成原来的形状
def transpose_output(X,num_heads):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)
```
### （）自注意力
```python
#自注意力机制本质上就是输入数据同时作为query,key,value，通过多头注意力模型寻找输入数据之间的相关性
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
attention.eval()
#batch_size,num_queries,num_kv_pairs,valid_lens,X,Y
batch_size, num_queries = 2, 4
valid_lens = torch.tensor([3,2])
X = torch.ones((batch_size,num_queries,num_hiddens))
attention(X,X,X,valid_lens).shape
```
### （）位置编码
```python
#位置编码，利用正弦以及余弦函数进行固定位置编码，分别针对偶数维度使用正弦函数，奇数维度使用余弦函数
#PositionalEncoding,self,num_hiddens,dropout,maxlen,P,X,dtype
class PositonalEncoding(nn.Module):
    #不需要引入**kwargs，因为函数的计算方法以及计算函数都是固定的，也不需要引入动态参数或不确定的参数
    def __init__(self, num_hiddens,dropout,max_len = 1000):
        super(PositonalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        #torch.zeros()传入的是一个矩阵的形状
        self.P = torch.zeros((1,max_len,num_hiddens))
        #位置编码本质上就是将时间步或是序列位置的索引值(maxlen)映射到特征维度(num_hiddens)上
        #通过不同频率的正弦和余弦函数，赋予模型在序列位置处理中的位置感知能力，以下是计算公式代码
        X = torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        #P的形状为(1,num_steps,num_hiddens)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)
    def forward(self,X):
        #通过:X.shape[1]将P的第二维与X的第二维即num_steps进行对齐，其他维度自动广播，然后将P放置在与X相同的设备上(GPU&CPU),否则无法进行计算
        X = X + self.P[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)
```
### （）前馈网络
```python
#建立基于位置的前馈网络，对输入的每个特征进行独立的映射，操作与位置无关，由两个全连接层以及一个激活层构成
#PositionWiseFFN,ffn_num_input,ffn_num_hiddens,ffn_num_outputs,dense1,relu,dense2
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,**kwargs):
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```
### （）层规范化
```python
#传入前馈网络处理后的数据(需要进行随机丢神经元dropout)以及输入数据，进行残差连接后进行层规范化
#AddNorm,normalized_shape,dropout
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    #使用Addnorm的前向传播时，传入上一层输入X以及当前层输出Y
    def forward(self, X, Y):
        return self.ln(self.dropout(Y)+X)
```
### （）编码器结构
```python
#构建transformer编码器模块，attention-> addnorm1-> ffn-> addnorm2  ,use_bias用于控制是否在层的计算中使用偏置项
#EncoderBlock,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias
#attention,addnorm1,ffn,addnorm2,X,valid_lens,Y
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias = False, **kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2 = AddNorm(norm_shape,dropout)
    def forward(self,X,valid_lens):
        Y = self.addnorm1(X,self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y,self.ffn(Y))

#构建transformer编码器，包括嵌入层->位置编码->num_layers个EncoderBlock模块
#相较于编码器模块，初始化参数多了vocab_size，用于将词表映射到隐藏层(特征维度)，num_layers，用于堆叠transformer模块
#TransformerEncoder,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias
#num_hiddens,embedding,pos_encoding,blks,'block'
class TransformerEncoder(d2l.Encoder):
    def __init__(self,vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias = False,**kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                            norm_shape, ffn_num_input, ffn_num_hiddens, 
                                                            num_heads, dropout, use_bias))
    #X,attention_weights,blks,i,blk,
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        #由于blks容器中存放的是(索引'block x'以及对应模型的键值对)，所以要想同时获得索引和模型，需要用enumerate进行遍历
        for i,blk in enumerate(self.blks):
            X = blk(X,valid_lens)
            #这个注意力权重的路径为blks->blk->d2l.MultiAttention->D2l.DotAttention->对应self.attention_weights
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
```
### （）解码器结构
```python
#构建解码器模块,注意训练阶段以及预推理阶段
#DecoderBlock,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens, num_heads,dropout, i,**kwargs
#attention1, addnorm1, attrntion2, addnorm2, ffn, addnorm3
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock,self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    #X, state, enc_outputs, enc_valid_lens, key_values, training, batch_size, num_steps, dec_valid_lens, X2, Y, Y2, Z
    def forward(self, X, state):#返回值包括规范化后的输出以及包含编码器输出，编码器有效长度以及解码器当前以及历史输入数据
        #利用保存当前解码器输入以及历史输出的隐状态state,来获取编码器输出(作为解码器输入)以及编码序列有效长度
        enc_outputs, enc_valid_lens = state[0], state[1]
        #state[2]实际上存放的是当前解码器输入块的历史输入信息，历史无数据，当前输入作为key_values,最后在传回state[2][self.i]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i],X), axis = 1)
        #注意key_values是包括解码器输入X以及历史数据，因此进行多头注意力训练时，可以视作自注意力机制
        state[2][self.i] = key_values
        #根据是否处于训练状态，训练时，需要根据输入X的形状(batch_size, num_steps, num_hiddens)对解码器有效长度
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        #自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        #多头注意力,利用解码器输出(Y)以及编码器输入(enc_outputs)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

#解码器构建，结构为嵌入层-> 位置编码-> num_layers层decoder_blk(blks)->全连接层
#TransformerDecoder, AttentionDecoder, vocab_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_outputs,ffn_num_hiddens
#num_heads, num_layers, dropout, **kwargs, embedding, pos_encoding, blks
class TransformerDecoder(d2l.AttentionDecoder):
    #不需要引入偏置项bias
    def __init__(self,vocab_size, key_size, query_size, value_size, 
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder,self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                                            norm_shape, ffn_num_input, ffn_num_hiddens, 
                                                            num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens,vocab_size)
    #enc_outputs, enc_valid_lens, *args
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    #X,attention_weights,blks,i,blk,
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        #注意力权重容器初始化，需要设置两组，一组用于存放自注意力权重，一组用于存放编码器解码器自注意力权重
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        #由于blks容器中存放的是(索引'block x'以及对应模型的键值对)，所以要想同时获得索引和模型，需要用enumerate进行遍历
        for i,blk in enumerate(self.blks):
            X, state = blk(X, state)
            #这个注意力权重的路径为blks->blk->d2l.MultiAttention->D2l.DotAttention->对应self.attention_weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    @property
    def attention_weights(self):
        return self._attention_weights
```
### （）模型参数初始化以及编码器解码器耦合
```python
#进行模型参数初始化
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
### （）训练流程
进行序列到序列训练，可以使用d2l封装好的训练模块d2l.train_seq2seq()
```python
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```
## 五.实验与结果分析 (Experiments and Results)
### 1.损失曲线
![loss_function_curve](loss_function_curve.png)
### 2.注意力权重热力图
- 编码器自注意力权重

![encoder_attention_weights](encoder_attention_weights.png)

- 解码器自注意力权重
  
![decoder_self_attention_weights](decoder_self_attention_weights.png)

- 编码器解码器注意力权重
  
![decoder_self_attention_weights](decoder_self_attention_weights.png)
### 3.测试结果展示(bleu)

loss 0.029, 5164.5 tokens/sec on cuda:0
| original sentence => target sentence       | bleu       | 
|------------|------------|
|go .=>va ! |  bleu 1.000|
|i lost .=>j'ai perdu .|  bleu 1.000|
|he's calm .=>il est calme .|  bleu 1.000|
|I'm home .=>je suis chez moi .|  bleu 1.000|

实验设计：简要描述你如何设计实验，包括数据集的选择、训练和评估方法。
结果展示：展示模型的训练过程、评估指标（如准确率、F1 分数、损失函数等），并通过图表展示模型的表现。如果有与其他模型的比较，可以展示对比结果。
模型优化：说明你如何优化模型，比如调整超参数、增加正则化、改变网络结构等，来提高性能。
## 六.技术挑战以及个人思考
### .分词以及词元索引转化
### .训练过程中张量数据的格式转换
### .参数列表匹配
### .动态模块化
### .模块化设计，封装技术细节

