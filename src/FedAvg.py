import os  
import copy  
import string  
from tqdm import tqdm 
import numpy as np  
import random
import torch  
from  torch.utils.data import Subset, DataLoader, random_split, ConcatDataset  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision  
import torchvision.transforms as transforms  
import args

# 引数を取得
args = args.get_args()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



#乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)   
    torch.manual_seed(seed)    
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False   

# デバイスの設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net().to(device)

# データセットの準備
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False)

num_classes = 10

# テストを行う関数
def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        running_loss = 0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            correct_preds += outputs.max(1)[1].eq(labels).sum().item()
            total_preds += outputs.size(0)

        loss = running_loss/len(test_loader)
        acc = 100*correct_preds/total_preds
    return loss, acc

def train(args, model, train_loader, device):
    # 最適化手法の定義
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()
    
    train_loss = 0.0

    # model 学習モードに設定
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # viewで1次元配列に変更
        images, labels = images.view(-1, 28*28).to(device), labels.to(device)
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
        # lossを累積
        train_loss += loss.item()
        
    train_loss = train_loss/len(train_loader)

    return train_loss

# diricletデータ分割
def split_dataset(dataset):
    # クラスごとにデータを分割
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (data, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 各クラスのデータをシャッフル
    for indices in class_indices.values():
        np.random.shuffle(indices)

    # Dirichlet分布に基づいてデータを各クライアントに割り当て
    client_indices = {i: [] for i in range(args.device)}
    for c, indices in class_indices.items():
        proportions = np.random.dirichlet(np.repeat(args.dirichlet, args.device))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        
        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)

    # 各クライアントのデータセットを作成
    subsets = [Subset(dataset, client_indices[i]) for i in range(args.device)]
    
    return subsets

# パラメータ集約
def average_models(client_updates):
    avg_state = copy.deepcopy(client_updates[0])
    for key in avg_state.keys():
        for i in range(1, len(client_updates)):
            avg_state[key] += client_updates[i][key]
        avg_state[key] = torch.div(avg_state[key], len(client_updates))
    return avg_state

# client選択
def choose_clients():
    return random.sample(range(0, args.device), args.cohort)

def fedetated_learning():
    # データセットを分割
    subsets = split_dataset(train_dataset)

    # 各クライアントのデータローダーを作成
    client_loaders = [DataLoader(subset, batch_size=args.bazhrtrhm tch_size, shuffle=True) for subset in subsets]

    global_model = Net().to(device)
    global_model.train()

    for round in range(args.round):
        print(f"Round {round+1}: {args.round}")

        client_updates = []

        for client_idx in choose_clients():
            local_model = copy.deepcopy(global_model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=args.lr)

            # ローカルでモデルを訓練
            train_loss = train(args, local_model, client_loaders[client_idx], device)
            print(f" Client {client_idx+1}, Train Loss: {train_loss:.4f}")

            client_updates.append(local_model.state_dict())

        # クライアントのモデルを平均化してグローバルモデルを更新
        avg_state = average_models(client_updates)
        global_model.load_state_dict(avg_state)

        # グローバルモデルを評価
        test_loss, test_acc = test(global_model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.2f}%\n")

    return global_model

if __name__ == "__main__":
    set_seed(42)
    fedetated_learning()
