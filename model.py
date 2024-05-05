import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        low_rank_modification = self.alpha * (x @ self.A @ self.B)
        return low_rank_modification

class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super(LinearWithLoRA, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x):
        original_out = self.linear(x)
        lora_out = self.lora(x)
        return original_out + lora_out

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes, rank, alpha):
        super(ModifiedResNet18, self).__init__()
        original_model = models.resnet18(pretrained=True)

        for param in original_model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = LinearWithLoRA(original_model.fc.in_features, num_classes, rank, alpha)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ModifiedConv2d(nn.Module):
    def __init__(self, conv_layer, rank, alpha):
        super(ModifiedConv2d, self).__init__()
        self.lora = LoRAConv(conv_layer, rank, alpha)

    def forward(self, x):
        return self.lora(x)

class LoRAConv(nn.Module):
    def __init__(self, conv_layer, rank, alpha):
        super(LoRAConv, self).__init__()
        self.alpha = alpha
        self.rank = rank
        self.conv_layer = conv_layer

        self.A = nn.Parameter(torch.randn(self.conv_layer.out_channels, self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, self.conv_layer.in_channels * self.conv_layer.kernel_size[0] * self.conv_layer.kernel_size[1]))

    def forward(self, x):
        low_rank_weights = self.alpha * (self.A @ self.B).view(self.conv_layer.out_channels, self.conv_layer.in_channels, *self.conv_layer.kernel_size)
        adapted_weights = self.conv_layer.weight + low_rank_weights
        return F.conv2d(x, adapted_weights, self.conv_layer.bias, stride=self.conv_layer.stride, padding=self.conv_layer.padding)

class ModifiedResNet18Conv(nn.Module):
    def __init__(self, num_classes, rank, alpha):
        super(ModifiedResNet18Conv, self).__init__()
        original_model = models.resnet18(pretrained=True)
        self.conv1 = ModifiedConv2d(original_model.conv1, rank, alpha)
        self.features = nn.Sequential(*list(original_model.children())[1:-1])
        self.fc = original_model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
