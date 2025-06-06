import torch
from torch import nn


class CNN(nn.Module):
    #input_size 是指输入序列的特征维度，hidden_size 是指 LSTM 单元中隐藏状态的维度，num_layers 是指 LSTM 模型中堆叠的 LSTM 层的数量，num_classes 是指分类任务的类别数量
    def __init__(self, seq_length,input_size, num_classes):
        super(CNN, self).__init__()

        self.cnn3_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
            nn.BatchNorm1d(input_size),
            nn.ELU(),  # activation
        )

        self.cnn3_2 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
            nn.BatchNorm1d(input_size),
        )

        self.cnn5_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
            nn.BatchNorm1d(input_size),
            nn.ELU(),  # activation
        )

        self.cnn5_2 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
        ),
            nn.BatchNorm1d(input_size),
        )

        self.cnn7_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
        ),
            nn.BatchNorm1d(input_size),
            nn.ELU(),  # activation
        )

        self.cnn7_2 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            # out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
        ),
            nn.BatchNorm1d(input_size),
        )



        self.cnn = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=int(input_size / 4),
            # out_channels = 1,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
            nn.BatchNorm1d(int(input_size / 4)),
            nn.ELU(),  # activation
        )

        self.fc1 = nn.Linear(int(input_size/4)*seq_length, int(input_size/4))
        self.fc2 = nn.Linear(int(input_size/4), num_classes)
        self.bn = nn.BatchNorm1d(input_size)
        #self.dropout = nn.Dropout(p=0.1)
        self.elu = nn.ELU()
        self.w = nn.Parameter(torch.ones(3)) # 4个分支, 每个分支设置一个自适应学习权重, 初始化为1
        #self.tanh = nn.Tanh()



    def forward(self, x):
        cnn_x = x.permute(0, 2, 1)  #(batch_size,50,1280) -> (batch-size,1280,50)

        #cnn_out1 = self.cnn1_1(cnn_x)
        #cnn_out1 = self.cnn1_2(cnn_out1)

        cnn_out3 = self.cnn3_1(cnn_x)
        cnn_out3 = self.cnn3_2(cnn_out3)

        cnn_out5 = self.cnn5_1(cnn_x)
        cnn_out5 = self.cnn5_2(cnn_out5)

        cnn_out7 = self.cnn7_1(cnn_x)
        cnn_out7 = self.cnn7_2(cnn_out7)

        #cnn_out1 = self.elu(cnn_out1 + cnn_x)
        cnn_out3 = self.elu(cnn_out3 + cnn_x)
        cnn_out5 = self.elu(cnn_out5 + cnn_x)
        cnn_out7 = self.elu(cnn_out7 + cnn_x)

        #cnn_out = torch.cat((cnn_out1,cnn_out3, cnn_out5), dim=1)

        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        #w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        #特征融合
        #cnn_out = w1 * cnn_out1 + w2 *cnn_out3 + w3 * cnn_out5
        cnn_out = self.bn(w1 * cnn_out3 + w2 * cnn_out5 + w3 * cnn_out7)

        cnn_out = self.cnn(cnn_out)
        view = cnn_out.view(cnn_out.size(0), -1)     #将网络展开(batch_size,320，length) -> (batchsize,320*length)  #进行了转置操作，转置会导致tensor不连续，而contiguous会重新深拷贝一个新的数据相同的tensor,否则无法进行view

        output = self.elu(self.fc1(view))
        output = self.fc2(output)

        return output