import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class backbone(nn.Module):
    def __init__(self,H,W):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    


class taskhead(nn.Module):
    def __init__(self,H,W):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,stride=1,padding=0)

        nn.init.normal_(self.conv1.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.conv1.bias,0)
        nn.init.normal_(self.conv2.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.conv2.bias,0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


    
class upsampling(nn.Module):
    def __init__(self,H,W):
        super().__init__()
        self.upsample = nn.Upsample(size=(H,W),mode='bicubic',align_corners=True)
    
    def forward(self, x):
        return self.upsample(x)


class CornerDetectionModel(nn.Module):
    
    def __init__(self, H, W):
        super().__init__()
        self.backbone = backbone(H, W)
        self.taskhead = taskhead(H, W)
        self.upsampling = upsampling(H, W)
    
    def forward(self, x):
        features = self.backbone(x)      
        detection = self.taskhead(features)  
        output = self.upsampling(detection)   
        return output


class criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, output, target):
        return self.bce_loss(output, target)

# 使用示例
if __name__ == "__main__":
    import torch
    
    # 创建模型
    num_epochs = 500000
    H, W = 256, 256  # 输入图像尺寸
    model = CornerDetectionModel(H, W)
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, H, W)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print("前向传播测试成功！")
    

    import torch.optim as optim
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()       