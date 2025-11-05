
import torch
import numpy as np

import pandas as pd
import utils as fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import time as tm
# system configuration
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

fn.set_seed(seed=2023, flag=True)

# hyper params
model_name = 'MDFANet'
#model_name = 'LSTM'
seq_l = 12
pre_l = 3
fea_l= 2
bs = 512
p_epoch = 200
n_epoch = 1000
is_train = True


start_time=tm.time()
# input data for ST-EVCDP
#occ, prc, adj, col, dis, cap, time, power, inf = fn.read_dataset()
#input data for UrbanEV
occ, prc, adj, col, dis, cap, time, power, inf = fn.read_dataset1()

adj_dense = torch.Tensor(adj)
adj_dense_cuda = adj_dense.to(device)
adj_sparse = adj_dense.to_sparse_coo().to(device)

# dataset division
train_occupancy,  test_occupancy = fn.division(occ, train_rate=0.8,  test_rate=0.2)
train_price, test_price = fn.division(prc, train_rate=0.8,  test_rate=0.2)

# data
train_dataset = fn.CreateDataset(train_occupancy, train_price, seq_l, pre_l, device, adj_dense)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)


test_dataset = fn.CreateDataset(test_occupancy, test_price, seq_l, pre_l, device, adj_dense)
test_loader = DataLoader(test_dataset, batch_size=len(test_occupancy), shuffle=False)


# training setting
model = models.MDFANet(a_sparse=adj_sparse).to(device)  # init model
# model = FGN().to(device)
#model = baselines.VAR().to(device)
#model = baselines.LSTM(seq_l, 2).to(device)
#model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda).to(device)
#model=baselines.GCN(seq_l,2,adj_dense_cuda).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
loss_function = torch.nn.MSELoss()
# 初始化最佳损失
best_loss = float('inf')

model.train()
for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
    train_loss = 0.0  # 初始化训练损失
    for j, data in enumerate(train_loader):
        model.train()
        occupancy, price, label = data

        optimizer.zero_grad()
        predict = model(occupancy, price)
        loss = loss_function(predict, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model, './checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(
                bs) + '_'  + '.pt')


model = torch.load('./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + '.pt',
                       weights_only=False)
# test
model.eval()
result_list = []
predict_list = np.zeros([1, adj_dense.shape[1]])
label_list = np.zeros([1, adj_dense.shape[1]])
for j, data in enumerate(test_loader):
    occupancy, price, label = data  # occupancy.shape = [batch, seq, node]
    print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
    with torch.no_grad():
        predict = model(occupancy, price)
        predict = predict.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        predict_list = np.concatenate((predict_list, predict), axis=0)
        label_list = np.concatenate((label_list, label), axis=0)

output_no_noise = fn.metrics(test_pre=predict_list[1:, :], test_real=label_list[1:, :])
result_list.append(output_no_noise)
result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'], data=result_list)
result_df.to_csv('./results' + '/' + model_name + '_' + str(pre_l) + 'bs' + str(bs) + '.csv', encoding='gbk')

# 记录结束时间
end_time = tm.time()

# 计算并打印总运行时间
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time:.2f}秒")


