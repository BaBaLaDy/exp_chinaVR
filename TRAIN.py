import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.optim as optim

class CustomCSVDataset(Dataset):
    def __init__(self, root_dir, transform=None, fixed_length=1000, included_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.data = []
        self.labels = []

        self.root = sorted(os.listdir(root_dir))  # 假设目录名即为类名
        # self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for subjects_idx, subjects in enumerate(self.root):
            subjects_dir = os.path.join(root_dir, subjects)

            if included_classes is not None:
                # for class_index, class_name in enumerate(os.listdir(subjects_dir)):
                for class_index, class_name in enumerate(included_classes):
                    user_dir = os.path.join(subjects_dir, class_name)
                    for csv_file in os.listdir(user_dir):
                        file_path = os.path.join(user_dir, csv_file)
                        # print(file_path)
                        data_frame = pd.read_csv(file_path, index_col=0, header=0, usecols=[0, 1, 2, 3])
                        self.data.append(data_frame.values)
                        self.labels.append(class_index)
                        # print(self.labels)
            else:
                for class_index, class_name in enumerate(os.listdir(subjects_dir)):
                    user_dir = os.path.join(subjects_dir, class_name)
                    for csv_file in os.listdir(user_dir):
                        file_path = os.path.join(user_dir, csv_file)
                        # print(file_path)
                        data_frame = pd.read_csv(file_path, index_col=0, header=0, usecols=[0, 1, 2, 3])
                        self.data.append(data_frame.values)
                        self.labels.append(class_index)
                        # print(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Truncate or pad the sample to fixed_length
        if len(sample) > self.fixed_length:
            sample = sample[:self.fixed_length]
        else:
            padding = np.zeros((self.fixed_length - len(sample), sample.shape[1]))
            sample = np.vstack((sample, padding))

        if self.transform:
            sample = self.transform(sample)

        # Reshape sample to (channels, height, width)
        sample = torch.tensor(sample, dtype=torch.float32).permute(1, 0)  # (features, sequence_length)
        sample = sample.unsqueeze(1)  # Add a channel dimension

        return sample, label


class ToTensorAndNormalize:
    def __call__(self, sample):
        # 合并通道
        sample = np.mean(sample, axis=1, keepdims=True)
        # 标准化
        sample = (sample - sample.mean()) / sample.std()
        return torch.tensor(sample, dtype=torch.float32)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),  # output size: (batch_size, 32, 600, 3)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),  # output size: (batch_size, 32, 600, 1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # output size: (batch_size, 32, 300, 3)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),  # output size: (batch_size, 64, 300, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # output size: (batch_size, 64, 150, 3)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),  # output size: (batch_size, 128, 150, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))  # output size: (batch_size, 128, 75, 3)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 75 * 1, 512),
            nn.ReLU(),
            # nn.Linear(512,256),
            # nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        return x


def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total



# Define model, loss function, and optimizer
# model = CustomCNN(num_classes=3, num_conv_layers=5, in_channels=3, hidden_channels=64)

if __name__ == '__main__':
    model = CustomCNN(num_classes=3)
    criterion = nn.CrossEntropyLoss()

    train_dataset0 = CustomCSVDataset(root_dir='datasets/Merge_fold_0713/ByNum_spline3', transform=ToTensorAndNormalize(), fixed_length=600, included_classes=['a1t', 'a7t','a8t'])
    train_dataset1 = CustomCSVDataset(root_dir='datasets/Merge_fold_0713/ByNum_spline4', transform=ToTensorAndNormalize(), fixed_length=600, included_classes=['a1t', 'a7t','a8t'])
    train_dataset2 = CustomCSVDataset(root_dir='datasets/Merge_fold_0713/ByNum_spline5', transform=ToTensorAndNormalize(), fixed_length=600, included_classes=['a1t', 'a7t','a8t'])

    finetune_dataset = CustomCSVDataset(root_dir='datasets/USC-HAD-clean', transform=ToTensorAndNormalize(), fixed_length=600, included_classes=['a1t', 'a7t','a8t'])
    test_dataset = CustomCSVDataset(root_dir='datasets/USC-HAD-clean', transform=ToTensorAndNormalize(), fixed_length=600, included_classes=['a1t', 'a7t','a8t'])

    # Create data loaders
    train_dataloader0 = DataLoader(train_dataset0, batch_size=5, shuffle=True)
    train_dataloader1 = DataLoader(train_dataset1, batch_size=5, shuffle=True)
    train_dataloader2 = DataLoader(train_dataset2, batch_size=5, shuffle=True)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    finetune_flag = False

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 20
    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader1:
            # for inputs, labels in finetune_dataloader:
            #     print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_accuracy = validate(model, train_dataloader0)
        test_accuracy = validate(model, test_dataloader)
        print(
            f"0Pretraining Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    #
    # Pretraining loop
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     for inputs, labels in train_dataloader1:
    #     # for inputs, labels in finetune_dataloader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         # print(outputs, labels)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #     train_accuracy = validate(model, train_dataloader1)
    #     test_accuracy = validate(model, test_dataloader)
    #     print(f"1Pretraining Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    #
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     for inputs, labels in train_dataloader2:
    #     # for inputs, labels in finetune_dataloader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         # print(outputs, labels)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #     train_accuracy = validate(model, train_dataloader2)
    #     test_accuracy = validate(model, test_dataloader)
    #     print(f"2Pretraining Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # # Fine-tuning loop
    if finetune_flag:
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        num_epochs = 10
        for epoch in range(num_epochs):
            for inputs, labels in finetune_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_accuracy = validate(model, finetune_dataloader)
            test_accuracy = validate(model, test_dataloader)
            print(
                f"Fine-tuning Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

