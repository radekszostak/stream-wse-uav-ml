import torch
import torch.nn as nn
import torchvision.models as models

class Vgg(nn.Module):
    def __init__(self, input_size, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.input_size = input_size
        ch = [64,128,256,512,256]
        fc_in_features = 16512
        affine = True
        momentum = 0.1
        track_running_stats=False
        # ch = [64,128,256,256,512]
        # fc_in_features = 32896
        # ch = [64,128,256,512,1024]
        # fc_in_features = 65664
        self.encoder = torch.nn.Sequential(
 
            # conv1
            torch.nn.Conv2d(2, ch[0], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[0], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[0], ch[0], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[0], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.MaxPool2d(2, stride=2),
            # conv2
            torch.nn.Conv2d(ch[0]+2, ch[1], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[1], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[1], ch[1], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[1], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.MaxPool2d(2, stride=2),
            # conv3
            torch.nn.Conv2d(ch[1]+2, ch[2], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[2], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[2], ch[2], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[2], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[2], ch[2], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[2], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.MaxPool2d(2, stride=2),
            # conv4
            torch.nn.Conv2d(ch[2]+2, ch[3], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[3], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[3], ch[3], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[3], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[3], ch[3], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[3], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.MaxPool2d(2, stride=2),
          
            #conv 5
            torch.nn.Conv2d(ch[3]+2, ch[4], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[4], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[4], ch[4], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[4], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.Conv2d(ch[4], ch[4], 3, padding=1),
            #torch.nn.BatchNorm2d(ch[4], affine=affine, momentum=momentum, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.MaxPool2d(2, stride=2),
        )
        
        
        
        fc_out_features = 1
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(fc_in_features,fc_out_features),
        )

    def forward(self, x):
        inp = x # for multiresolution
        tmp_size = self.input_size # for multiresolution
        for layer in self.encoder:
            if isinstance(layer, torch.nn.MaxPool2d):
                tmp_size = tmp_size//2 # for multiresolution
                x = layer(x)
                inp = torch.nn.functional.interpolate(inp,(tmp_size,tmp_size),mode='bilinear',align_corners=True) # for multiresolution
                x = torch.cat([inp,x],dim=1) # for multiresolution
            else:
                x = layer(x)
        x = torch.flatten(x,1,-1)
        #print(x.shape)
        x = self.dense(x)
        
        return x