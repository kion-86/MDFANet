import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha):
        super(MultiHeadsGATLayer, self).__init__()
        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=device))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=device))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1, device=device)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

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
        x = x.reshape(b * n, s)

        atts_stack = []
        for n_head in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n_head, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()
            atts = self.heads_dict[n_head, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features=in_channel, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class EMB(nn.Module):
    def __init__(self, time_channels, feature_channels, output_size):
        super(EMB, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1D1 = nn.Conv1d(time_channels, time_channels, kernel_size=3, padding=1)
        self.conv1D2 = nn.Conv1d(time_channels, time_channels, kernel_size=5, padding=2)
        self.conv1D_feature1 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.conv1D_feature2 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=time_channels, out_features=output_size)
        self.fc1 = nn.Linear(in_features=2 * time_channels, out_features=time_channels)
        self.fc2 = nn.Linear(in_features=time_channels, out_features=time_channels)
        self.fc3 = nn.Linear(in_features=feature_channels, out_features=feature_channels)

    def forward(self, x):
        time_conv = x
        time_conv1 = time_conv.permute(0, 2, 1)
        time_conv1 = self.conv1D1(time_conv1)
        time_conv1 = self.conv1D2(time_conv1)
        time_conv1 = time_conv1.permute(0, 2, 1)
        time_conv = torch.mul(time_conv, self.sigmoid(time_conv1))

        feature_conv = x.squeeze(1)
        feature_conv1 = self.conv1D_feature1(feature_conv)
        feature_conv1 = self.conv1D_feature2(feature_conv1)
        feature_conv = torch.mul(feature_conv, self.sigmoid(feature_conv1))

        in_feature = torch.add(time_conv, feature_conv)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)

        combined_data_cat = torch.cat([x, in_feature], -1)
        in_feature = F.gelu(self.fc1(combined_data_cat))
        in_feature = self.fc(in_feature)
        return in_feature


class MDFANet(nn.Module):
    def __init__(self, a_sparse, seq=12, m=2):
        super(MDFANet, self).__init__()
        self.feature = seq
        self.seq = seq
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        self.emb = EMB(time_channels=2, feature_channels=12, output_size=1)
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        self.fc1 = nn.Linear(in_features=2, out_features=1)
        self.fc2 = nn.Linear(in_features=self.seq, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = 2
        self.output_dim = 1
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.gru = nn.GRU(2, self.hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj2[i, i] = 0
        self.adj2 = adj2

    def forward(self, occ, prc):
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b * n, s, -1).unsqueeze(1)
        data = data.squeeze(1)

        data = self.emb(data)
        data = data.reshape(b, n, -1)

        atts_mat = self.gat_lyr(data)
        occ_conv1 = torch.matmul(atts_mat, data)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        atts_mat2 = self.gat_lyr(occ_conv1)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1

        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)

        output, _ = self.gru(x)
        output = self.output1(self.LeakyReLU(self.output0(output)))
        output = self.sigmoid(output)
        output = output.permute(0, 2, 1)
        output = self.fc2(output)
        output = output.view(b, n)

        return output