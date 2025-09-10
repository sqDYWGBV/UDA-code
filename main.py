import numpy as np
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything
from sklearn.neighbors import kneighbors_graph
from DSConv import DynamicSelfCorrectionConv1d
from AGCA import AGCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.utils import shuffle
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy
from DynamicSelfCorrectionConv2d import DynamicSelfCorrectionConv2d

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--source_data', type=str, default="data00/0HP/train_data.npy", help='')
    parser.add_argument('--source_label', type=str, default="data00/0HP/train_label.npy", help='')
    parser.add_argument('--target_data', type=str, default="data00/3HP/train_data.npy", help='')
    parser.add_argument('--target_label', type=str, default="data00/3HP/train_label.npy", help='')
    parser.add_argument('--source_testdata', type=str, default="data00/0HP/test_data.npy", help='')
    parser.add_argument('--source_testlabel', type=str, default="data00/0HP/test_label.npy", help='')
    parser.add_argument('--test_data', type=str, default="data00/3HP/test_data.npy", help='')
    parser.add_argument('--test_label', type=str, default="data00/3HP/test_label.npy", help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--nepoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='initialization list')
    args = parser.parse_args()
    return args

#定义数据集类
class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)

#定义加载数据函数
def load_data():
    source_data = np.load(args.source_data)
    print(source_data.shape)
    source_label = np.load(args.source_label).argmax(axis=-1)
    target_data = np.load(args.target_data)
    target_label = np.load(args.target_label).argmax(axis=-1)
    source_testdata = np.load(args.source_testdata)
    source_testlabel = np.load(args.source_testlabel).argmax(axis=-1)
    test_data = np.load(args.test_data)
    test_label = np.load(args.test_label).argmax(axis=-1)
    source_data = StandardScaler().fit_transform(source_data.T).T #进行归一化
    target_data = StandardScaler().fit_transform(target_data.T).T
    source_testdata = StandardScaler().fit_transform(source_testdata.T).T
    test_data = StandardScaler().fit_transform(test_data.T).T
    source_data = np.expand_dims(source_data, axis=1)  #增加一个批次维度
    target_data = np.expand_dims(target_data, axis=1)
    source_testdata = np.expand_dims(source_testdata, axis=1)
    test_data = np.expand_dims(test_data, axis=1)
    source_data, source_label = shuffle(source_data, source_label, random_state=2)
    target_data, target_label = shuffle(target_data, target_label, random_state=2)
    source_testdata, source_testlabel = shuffle(source_testdata, source_testlabel, random_state=2)
    test_data, test_label = shuffle(test_data, test_label, random_state=2)
    Train_source = Dataset(source_data, source_label)
    Train_target = Dataset(target_data, target_label)
    Test_source = Dataset(source_testdata, source_testlabel)
    Test_target = Dataset(test_data, test_label)
    print(Train_source)
    print(Test_source)
    return Train_source, Train_target, Test_source, Test_target


class Softmax(nn.Module):
    def __init__(self, m, n, source_output1, source_label,smoothing = 0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.m = torch.tensor([m]).to(self.device)
        self.n = torch.tensor([n]).to(self.device)
        self.source_output1 = source_output1.to(self.device)
        self.source_label = source_label.to(self.device)
        self.la, self.lb, self.lc, self.ld = [], [], [], []
        # Initialize data_set and label_set
        self.data_set = []
        self.label_set = []
        self.smoothing = smoothing
    def _combine(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a, self.b, self.c, self.d = [], [], [], []
        self.la, self.lb, self.lc, self.ld = [], [], [], []
        for i in range(self.source_label.size(0)):
            if self.source_label[i] == 0:
                self.a.append(self.source_output1[i])
                self.la.append(self.source_label[i].item())  # Save labels here
            elif self.source_label[i] == 1:
                self.b.append(self.source_output1[i])
                self.lb.append(self.source_label[i].item())
            elif self.source_label[i] == 2:
                self.c.append(self.source_output1[i])
                self.lc.append(self.source_label[i].item())
            elif self.source_label[i] == 3:
                self.d.append(self.source_output1[i])
                self.ld.append(self.source_label[i].item())

        if self.a:
            self.a = torch.stack(self.a).to(self.device)
        if self.b:
            self.b = torch.stack(self.b).to(self.device)
        if self.c:
            self.c = torch.stack(self.c).to(self.device)
        if self.d:
            self.d = torch.stack(self.d).to(self.device)

        # Use _class_angle to process each category
        a = self._class_angle(self.a, self.la)
        b = self._class_angle(self.b, self.lb)
        c = self._class_angle(self.c, self.lc)
        d = self._class_angle(self.d, self.ld)
        if isinstance(self.a, list):
            self.a = np.array(self.a)
        if isinstance(self.b, list):
            self.b = np.array(self.b)
        if isinstance(self.c, list):
            self.c = np.array(self.c)
        if isinstance(self.d, list):
            self.d = np.array(self.d)

        if len(a) != 0:
            self.data_set.append(a)
            self.label_set.append(torch.tensor(self.la).unsqueeze(1))
        if len(b) != 0:
            self.data_set.append(b)
            self.label_set.append(torch.tensor(self.lb).unsqueeze(1))
        if len(c) != 0:
            self.data_set.append(c)
            self.label_set.append(torch.tensor(self.lc).unsqueeze(1))
        if len(d) != 0:
            self.data_set.append(d)
            self.label_set.append(torch.tensor(self.ld).unsqueeze(1))
        if not (self.a.any() or self.b.any() or self.c.any() or self.d.any()):
            return torch.empty((0, self.source_output1.size(1))), torch.empty((0, 1))

        for i, tensor in enumerate(self.data_set):
            if tensor.dim() == 1:
                self.data_set[i] = tensor.unsqueeze(1).expand(-1, 4)  # 如果是一维张量，转换为二维并扩展为 [n, 4]
            elif tensor.size(1) == 1:
                self.data_set[i] = tensor.expand(-1, 4)  # 如果是 [n, 1] 的形状，扩展成 [n, 4]
        data = self.data_set[0]
        for i, tensor in enumerate(self.data_set):
            if tensor.dim() == 1:
                self.data_set[i] = tensor.unsqueeze(1).expand(-1, 4)  # 如果是一维张量，转换为二维并扩展为 [n, 4]
            elif tensor.size(1) == 1:
                self.data_set[i] = tensor.expand(-1, 4)  # 如果是 [n, 1] 的形状，扩展成 [n, 4]

            # 执行拼接操作
        data = self.data_set[0]
        for i in range(1, len(self.data_set)):
            if self.data_set[i].shape[1] != data.shape[1]:
                raise ValueError(
                    f"Tensor {i} shape {self.data_set[i].shape} does not match the expected shape {data.shape}")
            data = torch.cat([data, self.data_set[i]], dim=0)
        labels = torch.tensor(self.la + self.lb + self.lc + self.ld).view(-1, 1)
        labels = labels.to(device)
        if data.shape[0] != labels.shape[0]:
            # 调整数据或标签的批次大小，使其一致
            min_batch_size = min(data.shape[0], labels.shape[0])
            data = data[:min_batch_size]
            labels = labels[:min_batch_size]
        return data, labels.squeeze()
    def _class_angle(self, a, la):
        if len(la) == 0:
            return a
        else:
            index = la[0]
        for i in range(len(a)):
            c = a[i]
            part1 = c[:index]
            part2 = c[index + 1:]
            if c[index] > 0:
                val = c[index] / (self.m + 1e-5) - self.n
            elif c[index] <= 0:
                val = c[index] * (self.m + 1e-5) - self.n
            new_tensor = torch.cat((part1, val, part2)) if i == 0 else torch.vstack(
                [new_tensor, torch.cat((part1, val, part2))])
        return new_tensor

    def forward(self):
        data, label = self._combine()
        data = data.to(device)
        label = label.to(device)
        label_smoothed = self.label_smoothing(label,data.size(1))
        loss = F.kl_div(F.log_softmax(data,dim=1),label_smoothed,reduction='batchmean')
        return data, label, loss
    def label_smoothing(self, label, num_classes):
        device = label.device
        one_hot = torch.zeros(label.size(0), num_classes, device=device)
        label = label.unsqueeze(1)
        one_hot.scatter_(1, label, 1)  # 应该会将 one_hot 中 label 索引位置设置为 1
        return one_hot
class AdaBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaBN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    def forward(self, x, domain_stats=None):
        if self.training:
            # 训练模式，计算批数据的均值和方差
            batch_mean = x.mean([0, 2], keepdim=True)  # 适配1D卷积，计算沿着批次和时间步长的均值
            batch_var = x.var([0, 2], keepdim=True, unbiased=False)
            # 更新全局均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.view(-1)
            # 使用批数据的均值和方差
            mean, var = batch_mean, batch_var
        else:
            # 推理模式，使用传入的目标域统计信息（如果有）
            if domain_stats is not None:
                mean, var = domain_stats
            else:
                mean = self.running_mean.view(1, -1, 1)
                var = self.running_var.view(1, -1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)
        return x
from torch.nn import init
class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.A0 = torch.eye(hide_channel).to('cuda')
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(hide_channel, hide_channel, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(hide_channel, hide_channel, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, L = y.size()
        y = y.view(B, C, L)  # 确保 y 的形状正确
        # 计算 softmax
        y = self.softmax(self.conv2(y))
        # 计算 A1，确保形状正确
        y = y.view(B, C, L)
        # A1的生成方法
        A1 = y.view(B, C, L).transpose(1, 2)  # 对y进行转置操作，形状变为 [B, L, C]
        # A = self.A0 * A1 + self.A2，进行矩阵乘法前需要确保A1与A0的维度匹配
        A = torch.matmul(A1, self.A0) + self.A2  # 执行矩阵乘法
        # 调整形状并应用conv3，确保形状正确
        y = torch.matmul(A, y.view(B, C, L))  # 确保矩阵乘法形状匹配
        y = self.relu(self.conv3(y.view(B, C, 1)))
        y = y.view(B, C, 1)
        y = self.sigmoid(self.conv4(y))
        # 修改这里：将 y 扩展到与 x 形状相同
        y = y.expand_as(x)
        return x * y
class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.mutilconv3 = DynamicSelfCorrectionConv2d()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dsconv = DynamicSelfCorrectionConv1d().cuda()
        self.agca = AGCA(in_channel=64, ratio=4).cuda()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=64
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer,
                                                 num_layers=1)  # Number of layers in the transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=64
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=64
        )
        self.transformer1 = nn.Transformer(
            d_model=16,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=64
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dsconv(x1)
        x2 = x2.permute(2, 0, 1)
        print(x2.shape)
        memory = x2
        tgt = x2
        x3 = self.transformer1(src=x2, tgt=tgt)
        x3 = x3.permute(1, 2, 0)
        x4 = self.conv2(x3)
        x5 = self.mutilconv3(x4)
        # x6 = self.agca(x5)
        x7 = self.pool(x5)
        x7 = x7.view(x7.size(0), -1)
        return x7

class domain_adaptive(nn.Module):
    def __init__(self):
        super(domain_adaptive,self).__init__()
        self.fc = nn.Sequential(
                  nn.Linear(64, 16),
                  nn.ReLU(inplace=True),
                  nn.Linear(16, 4))
    def forward(self,x):
        x = self.fc(x)
        return x

def create_adjacency_matrix(features, n_neighbors=5):
    # 使用K近邻算法创建邻接矩阵
    features_np = features.detach().cpu().numpy()
    adjacency_matrix = kneighbors_graph(features_np, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    return adjacency_matrix
def coral_loss(source, target, adjacency_matrix):
    batch_size = source.size(0)
    d = source.size(1)
    source_cov = compute_weighted_covariance(source, adjacency_matrix)
    target_cov = compute_weighted_covariance(target, adjacency_matrix)
    loss = torch.sum(torch.pow((source_cov - target_cov), 2)) / (4 * d * d)
    return loss
def compute_weighted_covariance(features, adjacency_matrix):
    batch_size = features.size(0)
    features_centered = features - torch.mean(features, dim=0)
    if hasattr(adjacency_matrix, "toarray"):
        adjacency_matrix = adjacency_matrix.toarray()
    adjacency_matrix_cpu = adjacency_matrix.cpu()
    features_centered_cpu = features_centered.detach().cpu()
    weighted_features = adjacency_matrix_cpu @ features_centered_cpu.numpy()
    cov = torch.mm(weighted_features.clone().detach().T, weighted_features.clone().detach()) / (batch_size - 1)
    return cov
#定义LMMD
class LMMD(nn.Module):
    def __init__(self, m, n, num_kernels=5):
        super(LMMD, self).__init__()
        self.m = m
        self.n = n
        self.num_kernels = num_kernels

    def _mix_rbf_mmd2(self, X, Y, sigmas=(10,), wts=None):
        if wts is None:
            wts = [1] * len(sigmas)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            pairwise_distances_XX = torch.cdist(X, X, p=2)
            K_XX += wt * torch.exp(-gamma * pairwise_distances_XX ** 2)
            pairwise_distances_XY = torch.cdist(X, Y, p=2)
            K_XY += wt * torch.exp(-gamma * pairwise_distances_XY ** 2)
            pairwise_distances_YY = torch.cdist(Y, Y, p=2)
            K_YY += wt * torch.exp(-gamma * pairwise_distances_YY ** 2)
        return self._mmd2(K_XX, K_XY, K_YY)

    def _mmd2(self, K_XX, K_XY, K_YY):
        m = float(self.m)
        n = float(self.n)
        return torch.mean(K_XX) + torch.mean(K_YY) - 2.0 * torch.mean(K_XY)

    def lmmd(self, source, target, bandwidths=[1]):
        kernel_loss = self._mix_rbf_mmd2(source, target, sigmas=bandwidths) * 100.
        return kernel_loss

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.feature_extractor = feature_extractor()
        self.domain_adaptive = domain_adaptive()

    def forward(self,source_data, target_data):
        source_feature = self.feature_extractor(source_data)
        target_feature = self.feature_extractor(target_data)
        adjacency_matrix = create_adjacency_matrix(source_feature)
        source_representation = self.domain_adaptive(source_feature)
        target_representation = self.domain_adaptive(target_feature)
        return source_representation,target_representation,adjacency_matrix
losses = []
def train(model, epoch, source_loader, target_loader, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    lambd = 2 / (1 + np.exp(-10 * (epoch) / args.nepoch)) - 1
    model.to(device)
    for i in range(0, num_iter):
        source_data, source_label = next(iter_source)
        target_data, target_label = next(iter_target)
        source_data, source_label = source_data, source_label
        target_data, target_label = target_data, target_label
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        target_label = target_label.to(device)
        m = source_data.size()[0]
        n = target_data.size()[0]
        optimizer.zero_grad()
        output1, output2,adjacency_matrix = model(source_data.float(), target_data.float())  # , output2, target_data
        if isinstance(adjacency_matrix, torch.Tensor):
            if adjacency_matrix.is_cuda:
                print("Tensor is on GPU")
            else:
                print("Tensor is on CPU")
        else:
            adjacency_matrix = torch.tensor(adjacency_matrix.toarray()).to(device)
        data, label, clc_loss_step = Softmax(3, 0, output1, source_label).forward()#clc_loss_step为分类器损失
        pre_pseudo_label = torch.argmax(F.softmax(output2, dim=-1), dim=-1)#pre_pseudo_label，得到无标签目标域样本的伪标签
        pseudo_data, pseudo_label, pseudo_loss_step = Softmax(3, 0, output2, pre_pseudo_label).forward()
        C_loss = coral_loss(output1, output2,adjacency_matrix)
        L_loss = LMMD(m, n).lmmd(output1, output2)
        loss_step = clc_loss_step + (L_loss/(L_loss+C_loss))* L_loss + (C_loss/(L_loss+C_loss)) * C_loss + pseudo_loss_step
        loss_step.backward()
        optimizer.step()
        output1 = output1.to(device)
        source_label = source_label.to(device)
        output2 = output2.to(device)
        target_label = target_label.to(device)
        metric_accuracy_1.update(output1.max(1)[1], source_label)
        metric_accuracy_2.update(output2.max(1)[1], target_label)
        # metric_mean_1 = torchmetrics.MeanMetric().to(device)
        loss_step = loss_step.to(device)
        pseudo_loss_step = pseudo_loss_step.to(device)
        clc_loss_step = clc_loss_step.to(device)
        C_loss = C_loss.to(device)
        metric_mean_1.update(loss_step)
        metric_mean_2.update(criterion(output1, source_label))
        metric_mean_3.update(criterion(output2, target_label))
        metric_mean_4.update(pseudo_loss_step)
        metric_mean_5.update(clc_loss_step)
        metric_mean_6.update(C_loss)
        # metric_mean_7.update(MDA_loss)
    train_acc = metric_accuracy_1.compute()
    ###############################################
    test_acc = metric_accuracy_2.compute()
    #################################################
    train_all_loss = metric_mean_1.compute()  # loss_step
    train_loss = metric_mean_2.compute()
    test_loss = metric_mean_3.compute()
    target_cla_loss = metric_mean_4.compute()
    #################################################
    source_cla_loss = metric_mean_5.compute()
    #################################################
    cda_loss = metric_mean_6.compute()
    # mda_loss = metric_mean_7.compute()
    metric_accuracy_1.reset()
    metric_accuracy_2.reset()
    metric_mean_1.reset()
    metric_mean_2.reset()
    metric_mean_3.reset()
    metric_mean_4.reset()
    metric_mean_5.reset()
    metric_mean_6.reset()
    # metric_mean_7.reset()
    return train_acc, test_acc, train_all_loss, train_loss, test_loss, target_cla_loss, source_cla_loss, cda_loss
class Average:
    def __init__(self):
        self.total = 0.0
        self.count = 0
    def update(self, value, n=1):
        self.total += value * n
        self.count += n
    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count
    def reset(self):
        self.total = 0.0
        self.count = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test(model, source_loader, test_loader):
    model.eval()
    model = model.to(device)
    metric_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)  # 假设使用 Accuracy 指标进行评估
    metric_loss = Average()  # 假设使用平均损失指标进行评估
    iter_source = iter(source_loader)
    iter_test = iter(test_loader)
    num_iter = len(source_loader)
    all_predictions = []
    all_targets = []
    source_labels = []
    target_labels = []
    with torch.no_grad():
        for i in range(num_iter):
            try:
                source_data, source_label = next(iter_source)
            except StopIteration:
                iter_source = iter(source_loader)
                source_data, source_label = next(iter_source)
            try:
                test_data, test_label = next(iter_test)
            except StopIteration:
                iter_test = iter(test_loader)
                test_data, test_label = next(iter_test)

            source_data = source_data.float()  # 转换为float类型
            source_data = source_data.to(device)
            source_label = source_label.to(device)  # 注意是否需要传递标签数据
            test_data = test_data.float()  # 转换为float类型
            test_data = test_data.to(device)
            test_label = test_label.to(device)  # 注意是否需要传递标签数据
            # 执行推理
            output1, output2, _ = model(source_data, test_data)
            output1 = output1.to(device)
            output2 = output2.to(device)
            # 计算评估指标（如准确率和损失）
            acc = metric_accuracy(output2, test_label).to(device)
            loss = criterion(output2, test_label).to(device)
            metric_loss.update(loss.item(), len(test_label))

            # 记录预测结果和真实标签
            _, predicted = torch.max(output2, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(test_label.cpu().numpy())
            source_labels.extend(source_label.cpu().numpy())
            target_labels.extend(test_label.cpu().numpy())
    # 提取源域和目标域的特征向量
            source_features = output1.cpu().numpy()
            target_features = output2.cpu().numpy()
    # 计算评估指标的平均值
    test_acc = metric_accuracy.compute()
    test_loss = metric_loss.compute()
    # 重置度量器
    metric_accuracy.reset()
    metric_loss.reset()
    return test_acc, test_loss, conf_matrix


import torchmetrics
import time

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_accuracy_1 = torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)
    metric_accuracy_2 = torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)
    metric_mean_1 = MeanMetric().to(device)
    metric_mean_2 = MeanMetric().to(device)
    metric_mean_3 = MeanMetric().to(device)
    metric_mean_4 = MeanMetric().to(device)
    metric_mean_5 = MeanMetric().to(device)
    metric_mean_6 = MeanMetric().to(device)
    metric_mean_7 = MeanMetric().to(device)
    t_test_acc = 0.0
    stop = 0
    Train_source, Train_target,Test_source,Test_target = load_data()
    g = torch.Generator()
    source_loader = da.DataLoader(dataset=Train_source, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader = da.DataLoader(dataset=Train_target, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    source_testloader = da.DataLoader(dataset=Test_source, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    test_loader = da.DataLoader(dataset=Test_target, batch_size=args.batch_size, shuffle=True, generator=g)
    model = Model()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = 'best_model.pth'
    start_time = time.time()
    for epoch in range(0, args.nepoch):
        stop += 1
        train_acc, test_acc, train_all_loss, train_loss, test_loss, target_cla_loss, source_cla_loss, cda_loss = train(
            model, epoch, source_loader, target_loader, optimizer)
        print(
            'Epoch{}, train_loss is {:.5f},test_loss is {:.5f}, train_accuracy is {:.5f},test_accuracy is {:.5f},train_all_loss is {:.5f},target_cla_loss is {:.5f},source_cla_loss is {:.5f},cda_loss is {:.5f}'.format(
                epoch + 1, train_loss, test_loss, train_acc, test_acc, train_all_loss, target_cla_loss, source_cla_loss,
                cda_loss))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model at epoch {epoch + 1} with accuracy: {test_acc:.4f}")
    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练所用时间：{training_time} 秒")
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_loss, conf_matrix = test(model, source_testloader, test_loader)
    # 在最后加载最佳模
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    print("Test Loss: {:.4f}".format(test_loss))