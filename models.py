import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import math
from collections import OrderedDict
import re
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

__all__ = ['resnet18', 'resnet50', 'densenet121']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}

######################################################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)        
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine is not None:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)        

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)     
        
######################################################################   
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512): #512
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu: 
            add_block += [nn.ReLU()]
        if dropout: 
            add_block += [nn.Dropout(p=0.3)] 
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x    

######################################################################       
# SimCLRv2 Model using MixMatch ClassBlock(classifier),
# we leave first layer of projection head to be used in fine-tuning. (self.proj_head_used)
# Can either be based on resnet18, or resnet50.
# Set input size of self.proj_head_used as 512 for resnet18, 2048 for resnet50.
# Internal perceptron numbers can be changed for projection heads.

# This model implements model parallel, splitting its model into two GPU's.
# Then, it speeds up processing by pipelining inputs, and we can achieve concurrency
# since PyTorch launches CUDA operations asynchronously.
###################################################################### 
class MixSim_Model(nn.Module):
    def __init__(self, class_num, devices, dropout=0.2, split_size=10):
        super(MixSim_Model, self).__init__()

        fea_dim = 256
        self.dev0 = 'cuda:{}'.format(devices[0])
        self.dev1 = 'cuda:{}'.format(devices[-1])
        self.split_size = split_size

        model_ft = models.resnet18(pretrained=False)
        #model_ft = EfficientNet.from_name('efficientnet-b3', dropout_rate=dropout)

        self.model = model_ft

        # For ResNet
        self.seq1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,

            self.model.layer1).to(self.dev0)

        self.seq2 = nn.Sequential(
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool).to(self.dev1)

        # 512 for ResNet18, 2048 for ResNet50.
        self.proj_head_used = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True)).to(self.dev1)
        self.proj_head_disc = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True), nn.Linear(1024, fea_dim)).to(self.dev1)

        self.classifier = ClassBlock(512, class_num).to(self.dev1)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)

        # For ResNet
        s_prev = self.seq1(s_next).to(self.dev1)

        # For EfficientNet
        #s_prev = self.model.extract_features(s_next)
        #s_prev = self.model._avg_pooling(s_prev)
        #fea = torch.flatten(s_prev, 1).to(self.dev0)

        pre_l, pred_l = [], []

        # Pipelining inputs
        for s_next in splits:
            # Runs on dev0
            s_prev = self.seq2(s_prev)
            fea = torch.flatten(s_prev, 1)
            proj = self.proj_head_used(fea)
            pre_l.append(self.proj_head_disc(proj))
            pred_l.append(self.classifier(proj))

            # Runs on dev1
            s_prev = self.seq1(s_next).to(self.dev1)

            # For EfficientNet
            #s_prev = self.model.extract_features(s_next)
            #s_prev = self.model._avg_pooling(s_prev)
            #fea = torch.flatten(s_prev, 1).to(self.dev0)

        # Runs on dev0
        #proj = self.proj_head_used(fea)
        s_prev = self.seq2(s_prev)
        fea = torch.flatten(s_prev, 1)
        proj = self.proj_head_used(fea)
        pre_l.append(self.proj_head_disc(proj))
        pred_l.append(self.classifier(proj))

        return torch.cat(pre_l).to(self.dev0), torch.cat(pred_l).to(self.dev0)

######################################################################
# SimCLRv2 Model using MixMatch ClassBlock(classifier),
# we leave first layer of projection head to be used in fine-tuning. (self.proj_head_used)
# Can either be based on resnet18, or resnet50.
# Set input size of self.proj_head_used as 512 for resnet18, 2048 for resnet50.
# Internal perceptron numbers can be changed for projection heads.
######################################################################
class MixSim_Model_Single(nn.Module):
    def __init__(self, class_num, devices=[0], dropout=0.2):
        super(MixSim_Model_Single, self).__init__()

        fea_dim = 256

        model_ft = models.resnet18(pretrained=False)
        #model_ft = EfficientNet.from_name('efficientnet-b3', dropout_rate=dropout)

        self.model = model_ft
        self.proj_head_used = nn.Sequential(nn.Linear(512, 512),nn.ReLU(inplace=True))
        self.proj_head_disc = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), nn.Linear(512, fea_dim))

        self.classifier = ClassBlock(512, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # For Resnet
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        # For EfficientNet
        #x = self.model.extract_features(x)
        #x = self.model._avg_pooling(x)

        fea = torch.flatten(x, 1)

        # Layers used in unlabeled pre-training
        proj = self.proj_head_used(fea)
        pre = self.proj_head_disc(proj)

        # Classification model using half of projection head
        pred = self.classifier(proj)
        return pre, pred


######################################################################       
# Define the ResNet18-based Model
######################################################################     
class Res18_basic(nn.Module):
    def __init__(self, class_num):
        super(Res18_basic, self).__init__()
        fea_dim = 256
        model_ft = models.resnet18(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.fc_embed = nn.Linear(512, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(512, class_num)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        embed_fea = self.fc_embed(fea)
        pred = self.classifier(fea)
        return embed_fea, pred 

# Define the ResNet18-based Model
class Res18(nn.Module):
    def __init__(self, class_num):
        super(Res18, self).__init__()
        fea_dim = 256
        model_ft = models.resnet18(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.fc_embed = nn.Linear(512, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(512, class_num)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        embed_fea = self.fc_embed(fea)
        pred = self.classifier(fea)
        return embed_fea, pred 
    
        
class Res50(nn.Module):
    def __init__(self, class_num):
        super(Res50, self).__init__()
        fea_dim = 256        
        model_ft = models.resnet50(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()        
        self.model = model_ft
        self.fc_embed = nn.Linear(2048, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(2048, class_num)
        self.classifier.apply(weights_init_classifier)        
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        embed_fea = self.fc_embed(fea)
        pred = self.classifier(fea)
        return embed_fea, pred     
        
class Dense121(nn.Module):
    def __init__(self, class_num):
        super(Dense121, self).__init__()
        fea_dim = 256        
        model_ft = models.densenet121(pretrained=False)
        model_ft.features.classifier = nn.Sequential()
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.features.fc_embed = nn.Linear(1024, fea_dim)
        model_ft.features.fc_embed.apply(weights_init_classifier)  
        model_ft.classifier = ClassBlock(2048, class_num)
        model_ft.classifier.apply(weights_init_classifier)  
        self.model = model_ft
        
    def forward(self, x):
        x = self.model.features.conv0(x)
        x = self.model.features.norm0(x)
        x = self.model.features.relu0(x)
        x = self.model.features.pool0(x)
        x = self.model.features.denseblock1(x)
        x = self.model.features.transition1(x)
        x = self.model.features.denseblock2(x)
        x = self.model.features.transition2(x)
        x = self.model.features.denseblock3(x)
        x = self.model.features.transition3(x)
        x = self.model.features.denseblock4(x)
        x = self.model.features.norm5(x)
        x = self.model.features.avgpool(x)
        fea =  x.view(x.size(0), -1)
        embed_fea = self.model.features.fc_embed(fea)
        pred = self.model.classifier(fea)        
        return embed_fea, pred              
        

            
