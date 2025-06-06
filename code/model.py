import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, seq_length,input_size, num_classes):
        super(CNN, self).__init__()

        self.cnn3_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
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
            kernel_size=3,
            stride=1,
            padding=1,
        ),
            nn.BatchNorm1d(input_size),
        )

        self.cnn5_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
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
            kernel_size=5,
            stride=1,
            padding=2,
        ),
            nn.BatchNorm1d(input_size),
        )

        self.cnn7_1 = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
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
            kernel_size=7,
            stride=1,
            padding=3,
        ),
            nn.BatchNorm1d(input_size),
        )



        self.cnn = nn.Sequential(nn.Conv1d(
            in_channels=input_size,
            out_channels=int(input_size / 4),
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
        self.elu = nn.ELU()
        self.w = nn.Parameter(torch.ones(3))



    def forward(self, x):
        cnn_x = x.permute(0, 2, 1)

        cnn_out3 = self.cnn3_1(cnn_x)
        cnn_out3 = self.cnn3_2(cnn_out3)

        cnn_out5 = self.cnn5_1(cnn_x)
        cnn_out5 = self.cnn5_2(cnn_out5)

        cnn_out7 = self.cnn7_1(cnn_x)
        cnn_out7 = self.cnn7_2(cnn_out7)

        cnn_out3 = self.elu(cnn_out3 + cnn_x)
        cnn_out5 = self.elu(cnn_out5 + cnn_x)
        cnn_out7 = self.elu(cnn_out7 + cnn_x)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        cnn_out = self.bn(w1 * cnn_out3 + w2 * cnn_out5 + w3 * cnn_out7)

        cnn_out = self.cnn(cnn_out)
        view = cnn_out.view(cnn_out.size(0), -1)
        output = self.elu(self.fc1(view))
        output = self.fc2(output)

        return output