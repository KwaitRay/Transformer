import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AttentionDecoder(d2l.Decoder):
    #"""带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

#实现带有bahdanau注意力(attention_weights)的循环神经网络(隐状态state)
#Seq2SeqAttentionDecoder,AttentionDecoder,self,vocab_size,embed_size,num_hiddens,num_layers,dropout,**kwargs,attention,embedding,rnn,dense
#跟源代码一样，但是运行会有问题，复制过来后没问题
class Seq2SeqAttentionDecoder(AttentionDecoder):
    #分别对解码器的注意力，嵌入层，循环层，全连接层进行建模
    def __init__(self, vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqAttentionDecoder,self).__init__(**kwargs)
        #行业默认传入的key_size,value_size,num_hiddens大小一致，因此在d2l.AdditiveAttention中只需要传入num_hiddens参数
        self.attention = d2l.AdditiveAttention(num_hiddens,dropout)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)
    #enc_outputs,enc_valid_lens,outputs,hidden_state
    def init_state(self,enc_outputs,enc_valid_lens,*args):
        #outputs形状为(batch_size,num_steps,num_hiddens),permute之后为(num_steps,batch_size,num_hiddens)
        #hidden_state形状为(num_layers,batch_size,num_hiddens)
        outputs,hidden_state = enc_outputs
        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)
    #传入的X形状为(batch_size,num_steps),经过embedding以及permute就变成了(num_steps,batch_size,embed_size)
    #X,state,enc_outputs,hidden_state,enc_valid_lens,outputs,self._attention_weights,context
    def forward(self,X,state):
        #enc_outputs的形状为(batch_size,num_steps,num_hiddens),hidden_state形状为(num_layers,batch_size,num_hiddens)
        enc_outputs,hidden_state,enc_valid_lens =state
        #通过词嵌入，X形状多了一维embed_size
        X = self.embedding(X).permute(1,0,2)
        outputs,self._attention_weights=[],[]
        for x in X:
            #查询query通过获取hidden_state的最后一维，并在dim=1处插入一维（用于注意力机制中）
            #query和context形状都为(batch_size,1,num_hiddens)
            query =  torch.unsqueeze(hidden_state[-1],dim=1)
            context = self.attention(query,enc_outputs,enc_outputs,enc_valid_lens)
            x = torch.cat((context,torch.unsqueeze(x,dim=1)),dim=-1)
            #rnn模型前向传播返回的结果是输出值以及隐藏状态,outputs形状为(1,batch_size,num_hiddens)
            out,hidden_state = self.rnn(x.permute(1,0,2),hidden_state)
            #将每个时间步的输出以及权重矩阵传入outputs中
            outputs.append(out)
            #attention_weights的形状为(batch_size,1,num_steps),self_attention_weights是一个列表，列表中的每个元素即attention_weights
            self._attention_weights.append(self.attention.attention_weights)
        #将经过嵌入层，注意力层，以及循环层处理后的输出，在第一维度进行拼接，outputs变为(num_steps,batch_size,num_hiddens)
        #再经过全连接层进行处理nn.Linear(num_hiddens,vocab_size),形状变为(num_steps,batch_size,vocab_size)在返回值部分进行形状调整为(batch_size,num_steps,vocab_size)
        outputs = self.dense(torch.cat(outputs,dim=0))
        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]
    #!!!!!!!!!!!!!!!!!!!!@property将attention_weights方法设为只读，同时可以像访问属性一样访问，不需要method(),因此之后的step[0][0][0]才不需要()
    @property
    def attention_weights(self):
        return self._attention_weights

# class Seq2SeqAttentionDecoder(AttentionDecoder):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0, **kwargs):
#         super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
#         self.attention = d2l.AdditiveAttention(
#             num_hiddens, dropout)
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.GRU(
#             embed_size + num_hiddens, num_hiddens, num_layers,
#             dropout=dropout)
#         self.dense = nn.Linear(num_hiddens, vocab_size)

#     def init_state(self, enc_outputs, enc_valid_lens, *args):
#         # outputs的形状为(batch_size，num_steps，num_hiddens).
#         # hidden_state的形状为(num_layers，batch_size，num_hiddens)
#         outputs, hidden_state = enc_outputs
#         return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

#     def forward(self, X, state):
#         # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
#         # hidden_state的形状为(num_layers,batch_size,
#         # num_hiddens)
#         enc_outputs, hidden_state, enc_valid_lens = state
#         # 输出X的形状为(num_steps,batch_size,embed_size)
#         X = self.embedding(X).permute(1, 0, 2)
#         outputs, self._attention_weights = [], []
#         for x in X:
#             # query的形状为(batch_size,1,num_hiddens)
#             query = torch.unsqueeze(hidden_state[-1], dim=1)
#             # context的形状为(batch_size,1,num_hiddens)
#             context = self.attention(
#                 query, enc_outputs, enc_outputs, enc_valid_lens)
#             # 在特征维度上连结
#             x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
#             # 将x变形为(1,batch_size,embed_size+num_hiddens)
#             out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
#             outputs.append(out)
#             self._attention_weights.append(self.attention.attention_weights)
#         # 全连接层变换后，outputs的形状为
#         # (num_steps,batch_size,vocab_size)
#         outputs = self.dense(torch.cat(outputs, dim=0))
#         return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
#                                           enc_valid_lens]

#     @property
#     def attention_weights(self):
#         return self._attention_weights




#进行测试
#encoder,vocab_size,embed_size,num_hiddens,num_layers,decoder,X,dtype,state,output,state
encoder = d2l.Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size = 10, embed_size = 8,num_hiddens = 16,num_layers = 2 )
decoder.eval()
X = torch.zeros((4,7),dtype=torch.long)#传入X的形状为(batch_size,num_steps)
state = decoder.init_state(encoder(X),None)
output , state = decoder(X,state)
#output形状为(batch_size,num_steps,num_hiddens)，state包括三个部分,enc_outputs,hidden_state,enc_valid_lens
print('output.shape:',output.shape, 'len(state):',len(state),' state[0].shape:',state[0].shape,'len(state[1]):',len(state[1]),
    'state[1][0].shape:',state[1][0].shape)

#进行模型训练,先传入模型各层大小，训练集批次以及序列长度，模型参数(学习率，训练轮数，设备)
#embed_size,num_hiddens,num_layers,dropout ,batch_size,num_steps ,lr,num_epochs,device
#train_iter,src_vocab,tgt_vocab,net,engs,fras,eng,fra,translation,dec_attention_weight_seq
#net,eng,src_vocab,tgt_vocab,num_steps,device,True,attention_weights
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size,num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

#载入训练数据集，编码器，解码器以及组合成模型，进行训练
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout)
net = d2l.EncoderDecoder(encoder,decoder)
d2l.train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)



#根据训练好的模型，选取英语法语句子进行测试，根据模型进行预测
#如果传入的字符串中包含',需要用\表示这是字符串的一部分，而不是字符串结尾
engs = ['go .', 'i lost .', 'he\'s calm .', 'I\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs,fras):
    #利用d2l.predict_seq2seq来获取翻译结果以及解码器权重序列，然后打印翻译情况以及BELU结果，用于评估翻译效果，k=2表示参考两个词的组合的准确性
    translation,dec_attention_weight_seq = d2l.predict_seq2seq(net,eng,src_vocab,tgt_vocab,num_steps,device,True)
    print(f'{eng}=>{translation}, ',f'bleu {d2l.bleu(translation,fra,k=2):.3f}')
#获取解码器中的注意力权重，并转化为热力图能够解析的形式
#dec_attention_weight_seq中存储step的都是张量形式，step形状为(batch_size,num_heads,num_steps),step[0][0][0]获取的就是每个时间步的权重，然后进行拼接
attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
    1, 1, -1, num_steps))
#传入的矩阵需要进行切片操作，选取输入的最后一个句子engs[-1]，将该句子进行分词，然后+1(加上<eos>标记),然后将保存在gpu的张量转移到cpu中，不然matplotlib无法识别
d2l.show_heatmaps(attention_weights[:,:,:,:len(engs[-1].split())+1].cpu(),xlabel='Key_position',ylabel='Query_position')


plot.show()