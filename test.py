import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from config import config
from data_loader.dataset import DeepfakeDataset
from model.resnet3d import SDE_QNet
from utils import progress
from utils.seed import set_seed
import cv2
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

set_seed(config.SEED)
device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

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

def test():
    # 加载测试数据集（mode='test'一般不做数据增强）
    test_dataset = DeepfakeDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 构造模型，并加载训练过程中保存的最佳模型权重
    model = SDE_QNet(num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    model_name = get_model_name()
    model_path = os.path.join(config.MODEL_SAVE_PATH, model_name)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 测试集评估
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in progress.progress(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            logits, anomaly_scores = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
    pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, pred_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    print(f"测试集 ACC: {acc:.4f}, AUC: {auc:.4f}")

    
if __name__ == '__main__':
    test()
