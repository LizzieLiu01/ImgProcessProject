import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
import random
# ===============================
# Dummy OCR 数据集
# ===============================
class DummyOCRDataset(Dataset):
    def __init__(self, num_samples=200, imgH=32, imgW=100, num_classes=37):
        self.num_samples = num_samples
        self.imgH = imgH
        self.imgW = imgW
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(1, self.imgH, self.imgW)
        label_len = random.randint(3, 6)
        label = torch.randint(1, self.num_classes, (label_len,), dtype=torch.long)
        return img, label
# ===============================
# 1. CNN特征提取模块
# ===============================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv_layers(x)   # [B, 512, H', W']
        out = out.squeeze(2)        # 去掉高度维度 H'
        out = out.permute(2, 0, 1)  # [W', B, 512]  → 序列长度在前
        return out

# ===============================
# 2. RNN序列建模模块
# ===============================
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.embedding(recurrent)
        return output

# ===============================
# 3. CRNN整体模型
# ===============================
class CRNN(nn.Module):
    def __init__(self, imgH, nChannels, nHidden, nClasses):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "Image height must be multiple of 16"
        self.cnn = CNNFeatureExtractor()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, nClasses)
        )

    def forward(self, x):
        conv = self.cnn(x)
        output = self.rnn(conv)
        return output
####### 四 CTC Loss 训练流程
def ctc_loss(pred):

    criterion = nn.CTCLoss(blank=0)  # 0 代表空白符

    # 模型输出 [T, B, C]
    T, B, C = preds.size()
    preds_log_softmax = F.log_softmax(preds, dim=2)

    # 模拟标签 (例如识别 "AB12")
    targets = torch.tensor([1, 2, 27, 28], dtype=torch.long)
    target_lengths = torch.tensor([4])  # 每个样本长度
    input_lengths = torch.tensor([T])  # 每个输入序列长度

    loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
    print(loss)
##五、文字解码（推理阶段）
def ctc_decode(preds):
    preds_idx = preds.argmax(2)
    preds_idx = preds_idx.transpose(1, 0).contiguous().view(-1)
    # 去掉重复字符和空白
    chars = []
    prev = -1
    for i in preds_idx:
        if i != prev and i != 0:  # 0 为 blank
            chars.append(chr(i + 64))  # 简单映射：1->A, 2->B...
        prev = i
    return ''.join(chars)
def train():
    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 模拟数据
    dataset = DummyOCRDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            preds = model(imgs)
            preds_log_softmax = F.log_softmax(preds, dim=2)
            # ---- CTC长度参数 ----
            T = preds.size(0)  # 序列长度
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=T, dtype=torch.long)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
            # 将变长标签拼接成一维Tensor
            labels_concat = torch.cat(labels)
            loss = criterion(preds_log_softmax, labels_concat, input_lengths, target_lengths)

            # ---- 计算CTC损失 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            total_loss += loss.item()

def load_image_to_tensor(image_path, imgH=32):
    """
    读取本地图片，并转换为 CRNN 输入张量
    Args:
        image_path: 图片路径
        imgH: CRNN输入高度
    Returns:
        tensor: [1, 1, H, W] 灰度张量
    """
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    # 保持高宽比例缩放高度为imgH
    h, w = img.shape
    ratio = imgH / h
    new_w = int(w * ratio)
    img_resized = cv2.resize(img, (new_w, imgH))

    # 转为0~1浮点张量
    img_tensor = torch.from_numpy(img_resized).float() / 255.0

    # 添加通道和batch维度 [B, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    return img_tensor
# ===============================
# 4. 测试推理流程
# ===============================
if __name__ == '__main__':
    model = CRNN(imgH=32, nChannels=1, nHidden=256, nClasses=37)  # 26字母+10数字+空白
    print(model)

    # 假设输入灰度图片 [B, C, H, W] = [1, 1, 32, 100]
  #  input_data = torch.randn(1, 1, 32, 100)
    image_path = r"E:\workspace\doc\img\test.png"
    input_data = load_image_to_tensor(image_path, imgH=32)
    print("Input shape:", input_data.shape)  # [1,1,32,W]
    preds = model(input_data)
    ctc_loss(preds)
    train()
    print(preds.shape)  # [W', B, nClasses]




