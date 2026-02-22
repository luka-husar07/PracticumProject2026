import torch.nn as nn
import torch.nn.functional as F

# Building lightweight CNN model class

class SliceClassifier(nn.Module):
    def __init__(self, num_bins=5):
        super(SliceClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.4)  
        # Output for bin classification
        self.fc_bin = nn.Linear(256, num_bins) 
        # Output for plane classification  
        self.fc_plane = nn.Linear(256, 3)         

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        bin_output = self.fc_bin(x)     
        plane_output = self.fc_plane(x) 
        
        return bin_output, plane_output
