from torch import nn
import torch.nn.functional as F

# class de_obfuscator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # input size:
#         #   4 dims -> 3 dims
#         #   padding to (3, 128, 128)
#         #   input: (batch, 3, 128, 128)
#         self.conv1 = nn.Conv2d(8, 3, 5) # 8 kernels
#         self.conv2 = nn.Conv2d(8, 4, 3)
#         self.fc1 = nn.Linear(16*4*10, 30)
#         self.fc2 = nn.Linear(30, 2)

#     def forward(self, x):
#         # x : b, 16, 32, 32
#         input_size = x.size(0)
#         x = self.conv1(x) # b, 8, 24, 24 
#         x = F.relu(x)
#         x = F.avg_pool2d_pool2d(x, 2, 2) # b, 8, 12, 12
        
#         x = self.conv2(x) # b, 4, 10, 10
#         x = F.relu(x)
        
#         x = x.view(input_size, -1) # b, 4*10*10
        
#         x = self.fc1(x)
#         x = F.relu(x) # b, 30
        
#         x = self.fc2(x) # b, 2
#         x = F.relu(x)
        
#         x = F.log_softmax(x, dim=1)
        
#         return x

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}

# Alex -> de_obfuscator

class de_obfuscator(nn.Module):
    def __init__(self, num_classes=2):
        super(de_obfuscator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x