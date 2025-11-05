from zipfile import error

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as fn
import copy

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=device))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=device))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1, device=device)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse matrix
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        #print("1-mt_atts:", mt_atts.shape)
        mt_atts = self.linear(mt_atts)
        #print("2-mt_atts after linear:", mt_atts.shape)
        #print("3-self.values:", self.values.shape)
        new_values = self.values * mt_atts.squeeze()
        #print("4-new_values:", new_values.shape)
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        #print("5-atts_mat:", atts_mat.shape)
        atts_mat = atts_mat.to_dense() + self.mask
        #print("6-atts_mat:", atts_mat.shape)
        atts_mat = self.softmax(atts_mat)
        #print("7-atts_mat:", atts_mat.shape)
        return atts_mat


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features=in_channel, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=out_channel)
        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class EMB(nn.Module): #(EMB+CORE+mlp(x), plus feature dimentions,2*f)
    def __init__(self, time_channels, feature_channels, output_size):
        super(EMB, self).__init__()

        # 时间维度卷积层（捕获局部时间模式）  time_channels=2(feature number)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1D1 = nn.Conv1d(time_channels, time_channels, kernel_size=3, padding=1)  # 卷积核3
        self.conv1D2 = nn.Conv1d(time_channels, time_channels, kernel_size=5, padding=2)  # 卷积核5

        # 特征维度卷积层（捕获跨特征关系） feature_channels=12(time series period)
        self.conv1D_feature1 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, padding=1)  # 特征维卷积
        self.conv1D_feature2 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, padding=1)  # 特征维卷积

        # 全连接层
        self.fc = nn.Linear(in_features=time_channels, out_features=output_size)

        self.fc1 = nn.Linear(in_features=2 * time_channels, out_features=time_channels)
        self.fc2 = nn.Linear(in_features=time_channels, out_features=time_channels)
        self.fc3 = nn.Linear(in_features=feature_channels, out_features=feature_channels)

    def forward(self, x):
        # _, s, f = x.shape  # 假设输入的维度为 [batch_size*n, s, f]，分别是 [batch_size, 节点数, 时间步长, 特征数]

        # 时间维卷积
        time_conv = x  # [b*n,s,f]-->[b*n,s,f]
        time_conv1 = time_conv.permute(0, 2, 1)  # [b*n,s,f]-->[b*n,f,s]
        time_conv1 = self.conv1D1(time_conv1)  # [b*n,f,s]-->[b*n,f,s]
        time_conv1 = self.conv1D2(time_conv1)  # [b*n,f,s]-->[b*n,f,s]
        time_conv1 = time_conv1.permute(0, 2, 1)  # [b*n,f,s]-->[b*n,s,f]

        # 应用sigmoid激活函数，做元素级的乘法
        time_conv = torch.mul(time_conv, self.sigmoid(time_conv1))  # [b*n,s,f]
        # 应用softmax激活函数，做元素级的乘法
        #time_conv = torch.mul(time_conv, self.softmax(time_conv1))  # [b*n,s,f]
        # 特征维卷积
        feature_conv = x.squeeze(1)  # [b*n,1,s,f]-->[b*n,s,f]
        feature_conv1 = self.conv1D_feature1(feature_conv)  # [b*n,s,f]
        feature_conv1 = self.conv1D_feature2(feature_conv1)  # [b*n,s,f]

        # 应用sigmoid激活函数，做元素级的乘法
        feature_conv = torch.mul(feature_conv, self.sigmoid(feature_conv1))  # [b*n,s,f]
        # 应用softmax激活函数，做元素级的乘法
        #feature_conv = torch.mul(feature_conv, self.softmax(feature_conv1))  # [b*n,s,f]
        # 合并时间维卷积和特征维卷积
        in_feature = torch.add(time_conv, feature_conv)  # [b*n,s,f]
        #MLP(x)
        x=self.fc2(x); #[b*n,s,f]-->[b*n,s,f]
        x= x.permute(0, 2, 1)  #[b*n,s,f]-->[b*n,f,s]
        x=self.fc3(x); #[b*n,f,s]-->[b*n,f,s]
        x= x.permute(0, 2, 1)  #[b*n,f,s]-->[b*n,s,f]

        #core fusion
        combined_data_cat = torch.cat([x, in_feature], -1)  # [b*n,s,f]-->[b*n,s,2f]
        in_feature = F.gelu(self.fc1(combined_data_cat))  # [b*n,s,2f]-->[b*n,s,f]

        # 全连接层处理
        in_feature = self.fc(in_feature)  # [b*n,s,f]-->[b*n,s,1]

        # 将输出reshape为 [batch_size, n, -1]，即 [b, n, f]
        #y = in_feature.reshape(b, n, -1)  # [b*n,s, 1] -> [b, n, s]

        return in_feature



class MDFANet(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2,):
        super(MDFANet, self).__init__()
        self.feature = seq  # 保存序列长度 seq
        self.seq = seq  # 计算卷积操
        # 作后得到的序列长度 seq-kcnn+1，用于后续操作
        self.alpha = 0.5  # 平滑参数，决定残差连接的权重，值为0.5
        self.m = m  # 存储 LSTM 的隐藏单元数量
        self.a_sparse = a_sparse  # 存储输入的稀疏邻接矩阵 a_sparse
        self.nodes = a_sparse.shape[0]  # 记录节点的数量，即邻接矩阵的第一个维度的大小
        #EMB
        self.emb = EMB(time_channels=2, feature_channels=12, output_size=1)

        # GAT
        #self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        #self.conv2d = nn.Conv2d(1, 1, (3, 2),padding=(1,0))  # input.shape = [batch, channel, width, height]，生成[b*n,1,f,s]
        # 定义了一个2D卷积层 self.conv2d，输入输出通道数都为1，卷积核的大小为 (kcnn, 2)。这个层用于处理时间序列数据。
        # 举例：假设输入是 [batch=32, channel=1, width=10, height=2]，经过这个卷积层后，输出的维度会是 [32, 1, 9, 1]。
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)
        # self.gat_lyr：定义了一个多头图注意力层 (GAT layer)，该层接受稀疏邻接矩阵 a_sparse 作为输入，并输出基于图结构的特征表示。
        # self.gcn：定义了一个线性层 (全连接层)，用于在GAT之后进一步变换图卷积后的节点特征。


        # TPA
        #self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        # self.lstm：定义了一个双层的LSTM模块，用于处理序列数据的时间特征。
        self.fc1 = nn.Linear(in_features=2, out_features=1)
        self.fc2 = nn.Linear(in_features=self.seq, out_features=1)
        #self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        # self.fc1、self.fc2、self.fc3：这三个线性层用于时间卷积网络中的TPA模块，逐步将LSTM的输出转化为最终的预测值。
        self.decoder = nn.Linear(self.seq, 1)
        # self.decoder：定义了一个线性层，用于将最后的预测值解码成所需的输出格式。

        #GRU
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = 2
        self.output_dim = 1
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.gru = nn.GRU(2, self.hidden_dim, batch_first=True)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        # self.dropout：定义了一个Dropout层，用于防止过拟合，丢弃率为0.5。
        self.LeakyReLU = nn.LeakyReLU()
        # self.LeakyReLU：定义了一个带泄露的ReLU激活函数，用于在GAT层和其他层中引入非线性。

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        # adj1 和 adj2：从稀疏矩阵转为密集矩阵，并创建两个副本用于不同的处理。
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        # 代码循环修改了 adj1 和 adj2 矩阵的对角线元素，adj1 的对角线元素为一个非常小的值，而 adj2 的对角线元素为0。
        degree = 1.0 / (torch.sum(adj1, dim=0))
        # degree：计算每个节点的度，并取倒数，这个值将用于度矩阵。
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        # degree_matrix：生成一个度矩阵，它是节点度的对角矩阵，用于图卷积中的归一化。
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2
        # self.degree_matrix 和 self.adj2：保存上述计算得到的度矩阵和修改后的邻接矩阵。`

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape  # [512, 247, 12]
        data = torch.stack([occ, prc], dim=3).reshape(b * n, s, -1).unsqueeze(1)

        # [512*247, 1, 12, 2] --> [126464, 1, 12, 2]
        #[b*n,1,s]contact[b*n,1,s]-->[b*n,1,s,f],f=2(feature)
        #print("ori occ+prc data.shape is", data.shape)
        #encode
         #conv time dimention
        data= data.squeeze(1) #[b*n,1,s,f]-->[b*n,s,f]

        #without Embedding
        #data = self.output1(data)  # [b*n,s,f]-->[b*n,s,f]-->[b*n,s,1]


        #Embedding
        data = self.emb(data) #[b*n,s,f]-->[b*n,s,1]
        # 将输出reshape为 [batch_size, n, -1]，即 [b, n, f]
        # reshape for GAT
        data = data.reshape(b, n, -1)  # [b*n,s, 1] -> [b, n, s]
        #old embedding model
        #data = self.conv2d(data)  #
        #data = data.squeeze().reshape(b, n, -1)  # reshape(b, n, -1)

        # #embedding with star model
        # data = data.squeeze(1) #[b*n,1,s,f]-->[b*n,s,f]
        # data = data.permute(0, 2, 1) #[b*n,s,f]-->[b*n,f,s]
        # data = self.STAR(data) #[b*n,f,s]-->[b*n,f,s]
        # data = data.permute(0, 2, 1)  #[b*n,f,s]-->[b*n,s,f]
        # data = self.fc(data )  # [b*n,s,f]-->[b*n,s,1]
        # data=data.reshape(b, n, -1) #[512,247,12]

        # #without ST model
        # output = self.fc2(data)  # [b*n,1,s]-->[b*n,1,1]
        # output = output.view(b, n) #[b*n,1,1]-->[b,n]


        # First GAT layer
        atts_mat = self.gat_lyr(data)    #[b,n,s]
        # [n, n] --> [247, 247] (GAT输出的注意力矩阵)
        #print("after gat atts_mat.shape is", atts_mat.shape)

        occ_conv1 = torch.matmul(atts_mat, data)
        # [247,247] matmul [512,247,11] --> [512,247,11] (广播机制)
        #[n,n] matmul[b,n,s-kcnn+1]-->[b,n,s-kcnn+1]
        #print("after matmul atts_mat and data cc_conv1.shape is", occ_conv1.shape)

        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))
        # [512,247,11] --> [512,247,11] (self.gcn为Linear(11,11))
        #print("after dropout cc_conv1.shape is", occ_conv1.shape)

        # Second GAT layer
        atts_mat2 = self.gat_lyr(occ_conv1)
        # [247, 247] (同上)
        #print("after gat second layer atts_mat2.shape is", atts_mat2.shape)

        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        # [247,247] matmul [512,247,11] --> [512,247,11]
        # [n,n] matmul[b,n,s-kcnn+1]-->[b,n,s-kcnn+1]
        #print("after matmul atts_mat2 and occ_conv1 cc_conv2.shape is", occ_conv2.shape)

        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))
        # [512,247,11] --> [512,247,11] (self.gcn同上)
        #print("after dropout cc_conv2.shape is", occ_conv2.shape)

        # Residual connections
        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        # [512,247,11] + [512,247,11] --> [512,247,11]
        #print("after res occ_conv1.shape is:", occ_conv1.shape)

        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        # [512,247,11] + [512,247,11] --> [512,247,11]
        #print("after res occ_conv2.shape is:", occ_conv2.shape)
        #
        # Reshape for LSTM
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        # [512,247, 11] --> [126464, 11]
        #[b,n,s-kcnn+1]-->[b*n,s-kcnn+1]
        #print("after review occ_conv1.shape is:", occ_conv1.shape)

        occ_conv2 = occ_conv2.view(b * n, self.seq)
        # [512,247, 11] --> [126464, 11]
        # [b,n,s-kcnn+1]-->[b*n,s-kcnn+1]
        #print("after review occ_conv2.shape is:", occ_conv2.shape)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)
        # [126464, 11, 2] (新增维度拼接)
        #[b*n,s-kcnn+1]contact[b*n,s-kcnn+1]-->[b*n,s-kcnn+1,2]
        #print("new data out of conv1 and conv2 x.shape:", x.shape)
        #x=x.view(b,n,self.seq,-1)   #[b*n,t,f]-->[b,n,t,f]
        # #print("new data out of GAT x.shape:", x.shape)
        # #UnetSTF
        # #x=x.permute(0,2,1)  #[b*n,s,f]--->[b*n,f,s]
        # #star model with atctivation
        # #x1 = self.dropout(self.LeakyReLU(self.STAR(x)))
        #
        # #x1=self.STAR(x)  #[b*n,f,s]-->[b*n,f,s] only STAR
        # #print("new data out of UnetTSF x1.shape:", x1.shape)
        # #reshap of x1 for the MLP
        # #x1=x1.reshape(b*n,self.seq,-1) #[b*n,t,f]
        #
        #
        #
        #GRU

        output, _ = self.gru(x) #[b*n,s,f]-->[b*n,s,f]
        #without GRU
        #output=x
        output = self.output1(self.LeakyReLU(self.output0(output)))  # [b*n,s,f]-->[b*n,s,f]-->[b*n,s,1]
        output = self.sigmoid(output) #[b*n,s,1]-->[b*n,s,1]
        output = output.permute(0,2,1)  #[b*n,s,1]-->[b*n,1,s]
        output = self.fc2(output)  # [b*n,1,s]-->[b*n,1,1]
        output = output.view(b, n) #[b*n,1,1]-->[b,n]


        #without GRU
        # x1=x.permute(0,2,1)   #[b*n,f,s]-->[b*n,s,f]
        # #print("new data x1 out of reshape:", x1.shape)
        # x1=self.fc1(x1)  #[b*n,t,1]
        # x1=x1.permute(0,2,1) #[b*n,t,1]-->[b*n,1,t]
        # y=self.fc2(x1) #[b*n,1,t]-->[b*n,1,1]
        #
        # #y = self.fc3(hx)
        # # [126464,1,8] --> [126464,1,1] (fc3: Linear(8,1))
        # # print("after fc3 with hx, y.shape:", y.shape)
        #
        # y = y.view(b, n)
        # # [126464,1] --> [512,247] (恢复batch和node维度)
        # #print("after reshape y.shape:", y.shape)
        return output


