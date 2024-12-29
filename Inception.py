# Inception

import torch
import torch.nn as nn
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, ch1x1, kernel_size=3, padding=1)
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3reduce, kernel_size=1),  # 1x1
            nn.Conv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )
        
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5reduce, kernel_size=1, padding=1),  # 1x1
            nn.Conv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=1)
        )
        
        self.poolconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)  # 1x1
        )

    def forward(self, x):
        y1 = self.conv1x1(x)
        y2 = self.conv3x3(x)
        y3 = self.conv5x5(x)
        y4 = self.poolconv(x)

        # print(y1.size(), y2.size(), y3.size(), y4.size())
        outputs = torch.cat([y1, y2, y3, y4], dim=1)
        return outputs

class InceptionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)  # (64, 192, 28, 28)
       
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x) # [64, 1024, 7, 7])
        x = self.avgpool(x)  # [64, 1024, 7, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = datasets.CIFAR10(root = "../DLPractice/Data/", train = True, transform = transform, download = True)
    test_dataset = datasets.CIFAR10(root = "../DLPractice/Data/", train = False, transform = transform, download = True)
    dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True) # , pin_memory=True, collate_fn=lambda x: x
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = InceptionNet(num_classes=10).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)
    loss_func = nn.CrossEntropyLoss()
    epoch = 100

    for e in (range(epoch)):
        
        model.train()
        losses = []
        for i, loader_data in enumerate(tqdm(dataloader), 0):

            image, target = loader_data

            image = image.to("cuda")
            target = target.to("cuda")

            y = model(image)

            loss = loss_func(y, target)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = sum(losses)
        print(f"Epoch {epoch + 1} : loss {train_loss.item()}")  # gpu라서  item 넣어봄

    torch.save(model.state_dict(), 'model_weights.pth')
    plt.plot(losses)
    plt.show()
