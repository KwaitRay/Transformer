import torch
from d2l import torch as d2l
import math
from torch import nn
import matplotlib.pyplot as plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

#进行模型初始化，然后传入参数计算注意力，然后进行展示注意力汇聚情况
#num_hiddens,num_heads,attention,
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
attention.eval()
#batch_size,num_queries,num_kv_pairs,valid_lens,X,Y
batch_size, num_queries = 2, 4
num_kv_pairs, valid_lens = 6, torch.tensor([3,2])
X = torch.ones((batch_size,num_queries,num_hiddens))
Y = torch.ones((batch_size,num_kv_pairs,num_hiddens))
attention(X,Y,Y,valid_lens).shape

#自注意力机制本质上就是输入数据同时作为query,key,value，通过多头注意力模型寻找输入数据之间的相关性
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
attention.eval()
#batch_size,num_queries,num_kv_pairs,valid_lens,X,Y
batch_size, num_queries = 2, 4
valid_lens = torch.tensor([3,2])
X = torch.ones((batch_size,num_queries,num_hiddens))
attention(X,X,X,valid_lens).shape

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

#进行位置编码测试，选取(1,num_steps,num_hiddens)形状的张量作为输入，转换成结合特征信息(X)以及位置信息(P)的输入
#便于transformer模型捕捉语义信息以及上下文关系,初始化参数，将PostionnalEncoding进行实例化，分别获取对应的结合了位置编码的输出X以及位置编码P
#对P进行切片，选取P第一批次(0)后形状为(num_steps,num_hiddens),再选取6-9列进行绘图，x轴就为时间步索引torch.arange(num_steps)
#encoding_dim,num_steps,pos_encoding,X,P,x_label,figsize,legend
encoding_dim, num_steps = 32, 60
pos_encoding = PositonalEncoding(encoding_dim,0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1,num_steps,encoding_dim)))
#!!!!!注意：冒号
P = pos_encoding.P[:,:X.shape[1],:]
#注意传入的y轴值P[0,:,6:10]要进行转置.T,因为形状的第一维对应y轴值，第二维对应x轴值，即num_steps
d2l.plot(torch.arange(num_steps),P[0,:,6:10].T, xlabel='Row_position', figsize=(6,2.5),legend=["Col %d" % d for d in torch.arange(6,10)])
#编码交叉程度越高，说明这些编码上下文之间联系越紧密，可以帮助模型捕捉到长距离依赖关系


#绝对位置编码，将二进制进行转换
for i in range(8):
    print(f'{i}的二进制是{i:>03b}')
#将获取的位置编码张量P进行形状调整，绘制热力图
P = P[0,:,:].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P,xlabel='Column (encoding dimension)',ylabel='Row (position)', figsize=(3.5,4),cmap='Blues')
#根据热力图图像，频率随着维度增加而减小一个比例：1 / 10000^(2i/d)，其中 i 是当前维度，d 是总维度，维度较低(encoding_dimension)，波动幅度越大
plot.show()
