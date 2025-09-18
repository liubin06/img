import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

# 1. 加载预训练模型和图像预处理
model_path =r'D:\python\results\self\0.5_128_model_300.pth'
state_dict = torch.load(model_path)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model = Model()  # 先创建模型实例
model.load_state_dict(new_state_dict)
model.eval()  # 切换到评估模式

# 图像预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. 定义钩子函数获取特征图
feature_maps = {}


def get_feature_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()  # 保存特征图

    return hook


# 注册需要观察的层（选择典型卷积层）
target_layers = ['f.3.0.conv1', 'f.5.0.conv1', 'f.6.0.conv1']
for name, module in model.named_modules():
    if name in target_layers:
        module.register_forward_hook(get_feature_hook(name))

# 3. 加载并处理图像
image_path = r"D:\Deskstop\1.png"  # 替换为你的图像路径
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # 添加批次维度

# 4. 前向传播获取特征图
with torch.no_grad():  # 不计算梯度，加快速度
    model(input_tensor)

# 5. 可视化原始图像
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("原始图像")
plt.axis('off')
plt.show()

# 6. 可视化特征图（每个层显示8个通道）
for layer_name in target_layers:
    fm = feature_maps[layer_name]  # 形状: [1, 通道数, 高, 宽]
    print(f"{layer_name} 特征图形状: {fm.shape}")

    # 显示前8个通道
    plt.figure(figsize=(16, 2))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(fm[0, i], cmap='gray')  # 灰度显示
        plt.title(f"通道 {i}")
        plt.axis('off')
    plt.suptitle(f"层: {layer_name} 的特征图", y=1.05)
    plt.show()

# 7. 生成并显示热图（以最深层为例）
last_layer = target_layers[-1]
fm = feature_maps[last_layer]

# 计算特征图加权平均（简单版CAM）
weights = torch.mean(fm, dim=(2, 3), keepdim=True)  # 全局平均池化作为权重
heatmap = torch.sum(weights * fm, dim=1).squeeze()  # 加权求和
heatmap = torch.relu(heatmap) / torch.max(heatmap)  # 归一化

# 调整热图大小以匹配原图
from torchvision.transforms import Resize

heatmap_resized = Resize(img.size[::-1])(heatmap.unsqueeze(0).unsqueeze(0)).squeeze()

# 叠加显示
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(heatmap_resized.numpy(), cmap='jet', alpha=0.5)  # 叠加热图
plt.title(f"{last_layer} 关注区域热图")
plt.axis('off')
plt.show()
