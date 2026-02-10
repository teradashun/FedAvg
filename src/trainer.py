import torch
import torch.nn as nn


def train(optimizer, model, train_loader, device, model_name):
    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()

    # model 学習モードに設定
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if model_name == "DNN":
            # viewで1次元配列に変更
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        elif model_name == "MNIST_CNN" or "CIFAR10_CNN":
            images, labels = images.to(device), labels.to(device)
        
        # 勾配をリセット
        optimizer.zero_grad()
        # 推論
        outputs = model(images)
        # lossを計算
        loss = criterion(outputs, labels)
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()


def test(model, test_loader, device):

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            correct_preds += outputs.max(1)[1].eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    return acc