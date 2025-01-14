import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import math
import matplotlib.pyplot as plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

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
    
#测试,ffn,注意在ffn中传入的输入张量形状为(batch_size,num_steps,num_hiddens)最后一维要与初始化中的num_inputs相同，否则维度不匹配
ffn = PositionWiseFFN(4,4,8)
ffn.eval()
ffn(torch.ones((2,3,4)))[0]

#层规范化以及批量规范化，层规范化以样本为单位，对一个样本中所有特征维度进行规范化，批量规范化以特征为单位，针对所有样本的列（即单个特征维度）
#层规范化更适合于序列数据(RNN与NLP)以及小批量数据，批量规范化适合于CV任务以及大批量数据的任务
#ln,bn,X[[1,2],[2,3]],dtype
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1,2],[2,3]],dtype=torch.float32)
print('Layer norm:',ln(X),'Batch norm:',bn(X))

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
    
#测试,add_norm,一次规范化张量的后两个维度[3,4](输入张量的后两维要对应)，即同时对位置编码以及特征维度进行规范化
add_norm = AddNorm([3,4],0.5)
add_norm.eval()
add_norm(torch.ones((2,3,4)),torch.ones((2,3,4))).shape

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

#测试,valid_lens,encoder_blk
X = torch.ones((2,100,24))
valid_lens = torch.tensor([3,2])
encoder_block = EncoderBlock(24, 24, 24, 24,[100,24],24,48,8,0.5)
encoder_block.eval()
encoder_block(X,valid_lens).shape

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
    
#测试,encoder
encoder = TransformerEncoder(200, 24, 24, 24, 24, [100,24], 24, 48, 8, 2, 0.5)
encoder.eval()
#输入数据的形状为(batch_size,len_sequence)可以看作是句子数量以及单个句子的序列长度，vocab_size表示每个句子中的元素(单词)的索引范围，词汇表大小
encoder(torch.ones((2,100), dtype=torch.long),valid_lens).shape#注意要选用torch.long作为类型

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

#进行预训练，测试模型
#decoder_blk, X, state, 
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
#注意输入的后两维要与norm_shape对应，便于进行规范化
X = torch.ones((2,100,24))
state = [encoder_block(X, valid_lens), valid_lens, [None]]#传参用()
decoder_blk(X, state)[0].shape

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
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

#根据训练好的模型，选取英语法语句子进行测试，根据模型进行预测
#如果传入的字符串中包含',需要用\表示这是字符串的一部分，而不是字符串结尾
engs = ['go .', 'i lost .', 'he\'s calm .', 'I\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs,fras):
    #利用d2l.predict_seq2seq来获取翻译结果以及解码器权重序列，然后打印翻译情况以及BELU结果，用于评估翻译效果，k=2表示参考两个词的组合的准确性
    translation,dec_attention_weight_seq = d2l.predict_seq2seq(net,eng,src_vocab,tgt_vocab,num_steps,device,True)
    print(f'{eng}=>{translation}, ',f'bleu {d2l.bleu(translation,fra,k=2):.3f}')

#获取编码器自注意力权重，并进行可视化
#enc_attention_weights
enc_attention_weights = torch.cat(net.encoder.attention_weights,0).reshape((num_layers, num_heads, -1, num_steps))
d2l.show_heatmaps(enc_attention_weights.cpu(),xlabel='Key positions', ylabel='Query positions', titles=['Head %d'%i for i in range(1,5)],
                  figsize=(7, 3.5))

#分别获取解码器的自注意力权重以及编码-解码器注意力权重
#dec_attention_weights_2d, step, dec_attention_weight_seq, attn, blk, head, dec_attention_weights_filled, dec_attention_weights
#dec_self_attention_weights, dec_inter_attention_weights
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weight_seq 
                            for attn in step 
                            for blk in attn 
                            for head in blk]
dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
#进行解包赋值，分别提取重排后第 0 维上的两个部分：dec_self_attention_weights: 自注意力权重。dec_inter_attention_weights: 交互注意力权重
dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)
#绘制热力图
d2l.show_heatmaps(dec_self_attention_weights, xlabel='Key positions', ylabel='Query positions', 
                  titles=['head %d'%i for i in range(1,5)], figsize=(7.5, 3))
d2l.show_heatmaps(dec_inter_attention_weights, xlabel='Key positions', ylabel='Query positions', 
                  titles=['head %d'%i for i in range(1,5)], figsize=(7.5, 3))
plot.show()