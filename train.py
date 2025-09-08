import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

from config import config
from data_loader.dataset import DeepfakeDataset
from model.resnet3d import SDE_QNet
from utils import progress
from utils.seed import set_seed

set_seed(config.SEED)
device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')


def weighted_loss(logits, labels, anomaly_scores, alpha=0.3):
    """
    使用 anomaly_scores 对损失进行加权。
    """
    # 计算标准交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='mean')(logits, labels)
    
    # 计算 anomaly_scores 的平均值
    anomaly_scores = anomaly_scores.mean(dim=1)  # [B,] 求平均
    # anomaly_scores = anomaly_scores.max(dim=1)[0]  # [B,]
    
    # 加权损失
    weighted_loss = loss * (1 + alpha * anomaly_scores)
    return weighted_loss.mean()  # 确保 loss 是一个标量


def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_anomaly_scores = []  # 存储 anomaly_scores

    for inputs, labels in progress.progress(dataloader, desc=f"Epoch {epoch} Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, anomaly_scores = model(inputs)  # 输出 logits 和 anomaly_scores
        loss = weighted_loss(logits, labels, anomaly_scores, alpha=0.5)  # 使用 anomaly_scores 加权损失
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(probs)
        all_anomaly_scores.extend(anomaly_scores.detach().cpu().numpy())  # 存储 anomaly_scores

    epoch_loss = running_loss / len(dataloader.dataset)
    pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    return epoch_loss, acc, auc, all_anomaly_scores


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_anomaly_scores = []  # 存储 anomaly_scores

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits, anomaly_scores = model(inputs)  # 输出 logits 和 anomaly_scores
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
            all_anomaly_scores.extend(anomaly_scores.detach().cpu().numpy())  # 存储 anomaly_scores

    epoch_loss = running_loss / len(dataloader.dataset)
    pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    return epoch_loss, acc, auc, all_anomaly_scores


def get_model_name():
    """
    根据 config.TRAIN_DATASET_PATHS 中的键生成模型保存名称： 
    如果选中的数据集正好为 {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"},
    则返回 "FF++_best_model.pth"；否则，将所有键按字母排序后拼接后加上 "_best_model.pth"。
    """
    dataset_names = list(config.TRAIN_DATASET_PATHS.keys())
    target_set = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
    target_set_c40 = {"Deepfakes(c40)", "Face2Face(c40)", "FaceSwap(c40)", "NeuralTextures(c40)"}
    
    if set(dataset_names) == target_set:
        return "FF++_best_model.pth"
    elif set(dataset_names) == target_set_c40:
        return "FF++_c40_best_model.pth"
    else:
        sorted_names = sorted(dataset_names)
        return "_".join(sorted_names) + "_best_model.pth"


def main():
    # 数据加载
    train_dataset = DeepfakeDataset(mode='train', split='train')
    val_dataset = DeepfakeDataset(mode='train', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 构造模型
    model = SDE_QNet(num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_auc = 0.0
    patience = config.PATIENCE
    trigger_times = 0
    model_name = get_model_name()
    save_path = os.path.join(config.MODEL_SAVE_PATH, model_name)
    
    train_losses, train_accs, train_aucs = [], [], []
    val_losses, val_accs, val_aucs = [], [], []
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc, train_auc, train_anomaly_scores = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_auc, val_anomaly_scores = validate(model, val_loader, criterion)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, ACC: {val_acc:.4f}, AUC: {val_auc:.4f}, Time: {elapsed:.2f}s")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            trigger_times = 0
            if not os.path.exists(config.MODEL_SAVE_PATH):
                os.makedirs(config.MODEL_SAVE_PATH)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at: {save_path}")
        else:
            trigger_times += 1
            print(f"EarlyStopping trigger_times: {trigger_times}")
            if trigger_times >= patience:
                print("Early stopping triggered. Training terminated.")
                break


if __name__ == '__main__':
    main()
