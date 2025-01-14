import torch
from torch import nn
from d2l import torch as d2l
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plot
#掩蔽softmax操作，将超过有效长度的元素用一个非常大的负值替代(-1e6)
#masked_softmax,X,valid_lens,shape,
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
#[2,3]变成[2,2,3,3], [[1,3],[2,4]]变成[1,3,2,4]
print(masked_softmax(torch.rand(2,2,4),torch.tensor([2,3])))
print(masked_softmax(torch.rand(2,2,4),torch.tensor([[1,3],[2,4]])))

#进行加性注意力模型构建
#AddictiveAttention,self,key_size,query_size,num_hiddens,dropout,**kwargs,W_k,bias,W_q,W_v
#queries,keys,values,valid_lens,features,scores,attention_weights
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
#开始随机生成查询集，键-值，有效长度，进行AdditiveAttention实例化，然后计算前向传播，根据前向传播中计算的注意力矩阵绘制热力图
#queries(2,1,20),keys(2,10,2),values(2,10,4),valid_lens(2,),attention
queries,keys= torch.normal(0,1,(2,1,20)),torch.ones((2,10,2))
#torch.float()不能直接float，要用dtype = torch,float32
values = torch.arange(40,dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
valid_lens = torch.tensor([2,6])
attention = AddictiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries,keys,values,valid_lens)
#reshape内部传入的是要转换成的形状，要用()关上
d2l.show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),xlabel='Keys',ylabel='Queries')

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
    
#构建输入数据，进行测试
#queries(2,1,2),attention
queries = torch.normal(0,1,(2,1,2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries,keys,values,valid_lens)
d2l.show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),xlabel='Keys',ylabel='Queries')
plot.show()

