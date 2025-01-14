import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib as plt
import matplotlib.pyplot as plot
#命名 num_rows,num_cols,fig,axes,row_axes,row_matrices,ax,matrix,pcm
#绘制注意力机制热图,传入的matrices的维度分别为行数，列数，查询数(自主性提示)，键的数目
#设置绘制图像格式，获取绘制热图行列数，遍历行列，分别绘制子图，如果是标题，设置标题，是第一列，设置y轴标签，第num_rows-1行，设置x轴标签
def show_heatmaps(matrices,xlabel,ylabel,titles = None,figsize = (2.5,2.5),cmap = 'Reds'):
    d2l.use_svg_display()
    num_rows ,num_cols= matrices.shape[0],matrices.shape[1]
    #d2l.plt.subplot()参数包括行列数，图像大小，是否共享x轴，y轴，以及返回的行列数组是否默认为二维
    fig , axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize,sharex =True,sharey =True,squeeze = False )
    for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
        for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(),cmap = cmap)
            if i == num_rows -1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                #不能直接传递titles，要传递titles[j]
                ax.set_title(titles[j])
    #在图像旁边添加一个colorbar,用于显示颜色与数值的映射关系，shrink=0.6表示缩放比
    fig.colorbar(pcm,ax=axes,shrink = 0.6)
#torch.eye(10)生成单位矩阵，10表示生成矩阵形状为(10,10),然后重新调整形状，attention_weights常用于表示掩码矩阵，用于保留对角线元素，每一头的注意力集中在对角线上
attention_weights = torch.eye(10).reshape((1,1,10,10))
show_heatmaps(attention_weights,xlabel='Keys',ylabel='Queries')


#nadaraya_watson_regression核回归
#生成训练样本和测试样本并排序
n_train = 50
#t为了更好的可视化注意力模式，需要进行排序torch.sort()有两个返回值，一个是返回按升序排序好的数组，一个是索引indices
# torch.rand(n_train)生成数量为n_train，(0,1)之间的随机数值_表示占位符，说明知道返回值有两个参数，但是不需要后一个参数(indices)
x_train,_ = torch.sort(torch.rand(n_train)*5)
#生成人工数据集，公式如下(别写错了)，然后在测试集上应用并添加噪声项，在训练集上应用当作生成实际结果
def f(x):
    return 2*torch.sin(x)+x**0.8
y_train = f(x_train)+torch.normal(0.0,0.5,(n_train,))
#测试集随机生成范围(0,5)步长为0.1的测试集,要是0.01图像x轴序列就太少了
x_test = torch.arange(0,5,0.1)
y_truth = f(x_test)
n_test = len(x_test)
#绘制图像，包括传入数据集x轴，y轴元素，'x','y',标签legend以及x轴y轴范围
def plot_kernel_reg(y_hat):
    d2l.plot(x_test,[y_truth,y_hat],'x','y',legend=['Truth','Pred'],xlim =[0,5],ylim=[-1,5])
    d2l.plt.plot(x_train,y_train,'o',alpha=0.5)
#torch.repeat_interleave()表示将参数一(y_train.mean()可以是数值，也可以是张量)重复n_test次，最后生成一个预测值序列，长度为n_test,形状为一维张量
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

#非参数注意力汇聚
#x_test->X_repeat(是测试集即对应查询重复后的结果),利用测试集-训练集等进行权重计算，然后将权重放在预测结果上，绘制图像
X_repeat = x_test.repeat_interleave(n_train).reshape((-1,n_train))
attention_weights = nn.functional.softmax(-(X_repeat-x_train)**2/2,dim = 1)
y_hat = torch.matmul(attention_weights,y_train)
plot_kernel_reg(y_hat)

#根据注意力权重张量，调整格式后绘制注意力权重热力图,传入参数为(行数，列数，查询数，键数)，因此需要unsqueeze(0)第0维前添加一维
#因为 attention_weights 的形状是 (n_test, n_train)，即每一行对应一个测试样本，列对应训练样本，所以热力图的 X 轴代表了训练样本的排序
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),xlabel='Sorted training inputs',ylabel='Sorted testing inputs')

#测试批次矩阵乘法torch.bmm(X,Y),对应的参数X，Y形状为(批次数，0维，1维)
X = torch.ones((2,1,4))
Y = torch.ones((2,4,6))
torch.bmm(X,Y).shape
weights = torch.ones((2,10))*0.1
#torch.arange(20.0)在没有指定范围和步长，会生成0-19的所有整数元素，然后进行reshape
values  = torch.arange(20.0).reshape((2,10))
#print(torch.bmm(weights.unsqueeze(1),values.unsqueeze(-1)))
torch.bmm(weights.unsqueeze(1),values.unsqueeze(-1))

#构建模型，包括模型初始化参数以及模型的而前向传播函数
#NWKernelRegression,__init__,**kwargs,requires_grad,w,queries,attention_weights
class NWKernelRegression(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.w=nn.Parameter(torch.rand((1,),requires_grad=True))
    def forward(self,queries,keys,values): 
        #由于需要的queries的形状与keys相同,为(n_test,n_train)即（查询个数，键值对个数）
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-self.w*(queries-keys)**2/2,dim=1)
        #values的形状为(查询个数，"键-值"对个数)，前向传播函数中的返回值就是最后的运行模型会得到的结果,形状为(n_test,1,1)或者(n_train,1,1)
        # 取决于在训练还是测试阶段，传入的数据集是训练集还是测试集
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)

#构造训练数据集，生成键值对
#表示将训练集输入沿第零维（1），进行重复n_train次，因此X_tile形状为(n_train,n_train)
X_tile = x_train.repeat((n_train,1))   
Y_tile = y_train.repeat((n_train,1))
#利用1-torch.eye(n_train)设置掩码，转换为torch.bool后将除了当前样本外的输入/输出（键/值）进行提取，然后调整形状为(n_train,n_train-1)
keys = X_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1)) 
values = Y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1)) 

#进行模型训练，梯度清除，前向传播，损失函数，后向传播，迭代，
#net,l,trainer,animator,epoch,
net = NWKernelRegression()
#none要小写
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(),lr = 0.5)
animator = d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[1,5])
for epoch in range(5):
    trainer.zero_grad()
    #训练时 queries 是 x_train，而 keys 和 values 也是基于 x_train 和 y_train 创建的,x_test是在测试时使用的
    l =  loss(net(x_train,keys,values),y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch:{epoch+1},loss:{float(l.sum()):.6f}')
    animator.add(epoch+1,float(l.sum()))

#进行测试
#keys,x_train,values,y_train,y_hat
#repeat()是操作整个张量，需要对张量的每个维度进行处理，而repeat_interleave(n,dim=xx)可以指定维度进行处理
keys = x_train.repeat((n_test,1))
values = y_train.repeat((n_test,1))
#根据前向传播函数，发现得到的y_hat形状为(n_test,)，因此需要在第二维前插入一个维度，之后从计算图中分离张量，使得张量的梯度计算停止
y_hat = net(x_test,keys,values).unsqueeze(1).detach()
#绘制预测值与实际值之间拟合图像，使用#nadaraya_watson_regression核回归
plot_kernel_reg(y_hat)
#绘制net中权重矩阵热力图,注意传入张量要是四维的，(行数，列数，查询数，键数)
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),xlabel='Sorted training inputs',ylabel='Sorted testing inputs')
plot.show()