import os

NORMALIZATION_STATS = {
    'Deepfakes': {'mean': [0.5321, 0.4181, 0.3980], 'std': [0.2643, 0.2185, 0.2210]},
    'Face2Face': {'mean': [0.5349, 0.4199, 0.4004], 'std': [0.2640, 0.2185, 0.2221]}, 
    'FaceSwap': {'mean': [0.5347, 0.4204, 0.4001], 'std': [0.2633, 0.2170, 0.2199]},
    'NeuralTextures': {'mean': [0.5349, 0.4185, 0.3985], 'std': [0.2650, 0.2191, 0.2218]},
    'CDF': {'mean': [0.5299, 0.3515, 0.3123], 'std': [0.2199, 0.1607, 0.1541]},
    'DFDC': {'mean': [0.4561, 0.3350, 0.3183], 'std': [0.2112, 0.1869, 0.1850]},
    'DFDC_P': {'mean': [0.4259, 0.3267, 0.2841], 'std': [0.2017, 0.1846, 0.1667]}
}

# 数据集路径（请根据实际情况修改）
TRAIN_DATASET_PATHS = {
    'Deepfakes': r'F:\STAR\dataSet\DataSets_video\Deepfakes',
    # 'Face2Face': r'F:\STAR\dataSet\DataSets_video\Face2Face',
    # 'FaceSwap': r'F:\STAR\dataSet\DataSets_video\FaceSwap',
    # 'NeuralTextures': r'F:\STAR\dataSet\DataSets_video\NeuralTextures',
    # 'Deepfakes(c40)': r'F:\STAR\dataSet\DataSets_video\Deepfakes(c40)',
    # 'Face2Face(c40)': r'F:\STAR\dataSet\DataSets_video\Face2Face(c40)',
    # 'FaceSwap(c40)': r'F:\STAR\dataSet\DataSets_video\FaceSwap(c40)',
    # 'NeuralTextures(c40)': r'F:\STAR\dataSet\DataSets_video\NeuralTextures(c40)',
    # 'CDF': r'F:\STAR\dataSet\DataSets_video\CDF',
    # 'DFDC': r'F:\STAR\dataSet\DataSets_video\DFDC',
    # 'DFDC_P':r'F:\STAR\dataSet\DataSets_video\DFDC_P'
}

TEST_DATASET_PATHS = {
    # 'Deepfakes': r'F:\STAR\dataSet\DataSets_video\Deepfakes',
    # 'Face2Face': r'F:\STAR\dataSet\DataSets_video\Face2Face',
    # 'FaceSwap': r'F:\STAR\dataSet\DataSets_video\FaceSwap',
    # 'NeuralTextures': r'F:\STAR\dataSet\DataSets_video\NeuralTextures',
    # 'Deepfakes(c40)': r'F:\STAR\dataSet\DataSets_video\Deepfakes(c40)',
    # 'Face2Face(c40)': r'F:\STAR\dataSet\DataSets_video\Face2Face(c40)',
    # 'FaceSwap(c40)': r'F:\STAR\dataSet\DataSets_video\FaceSwap(c40)',
    # 'NeuralTextures(c40)': r'F:\STAR\dataSet\DataSets_video\NeuralTextures(c40)',
    # 'CDF': r'F:\STAR\dataSet\DataSets_video\CDF',
    # 'DFDC':r'F:\STAR\dataSet\DataSets_video\DFDC',
    # 'DFDC_P':r'F:\STAR\dataSet\DataSets_video\DFDC_P',
    'Deepfakes(c50)': r'F:\STAR\dataSet\DataSets_video\Deepfakes(c50)',
    # 'CDF(c50)':r'F:\STAR\dataSet\DataSets_video\CDF(c50)',
    # 'DFDC(c50)':r'F:\STAR\dataSet\DataSets_video\DFDC(c50)'
}
TRAIN_VAL_SPLIT_RATIO = 0.8

# 训练超参数
NUM_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 模型参数
NUM_CLASSES = 2
NUM_FRAMES = 32

# 设备配置
DEVICE = 'cuda'

# 数据增强参数
TRAIN_TRANSFORM = {
    'resize': (112, 112),
}
TEST_TRANSFORM = {
    'resize': (112, 112),
}

# 模型保存路径
MODEL_SAVE_PATH = './checkpoints'
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
    print(f"模型保存路径已创建：{MODEL_SAVE_PATH}")

# 随机种子
SEED = 42

# 早停参数
PATIENCE = 5