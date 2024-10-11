import torch
from .BotNet.BotNetBottlent import *
from Config import load_config
from torchvision.models import resnet34,resnet18,vgg16,densenet121

args = load_config()

def Check_model(arg):
    if arg.model == 'Botnet18':
        model = Botnet18(pretrained=arg.PreTrain, NumClass=arg.numclass)
        print('Model use:->> Botnet18')

    elif arg.model == 'Botnet34':
        model = Botnet34(pretrained=arg.PreTrain, NumClass=arg.numclass)
        print('Model use:->> Botnet34')

    elif arg.model == 'ResNet18':
        model = resnet18(pretrained=arg.PreTrain)
        model.fc = nn.Linear(in_features=512, out_features=arg.numclass, bias=True)
        print('Model use:->> ResNet18')

    elif arg.model == 'ResNet34':
        model = resnet34(pretrained=arg.PreTrain)
        model.fc = nn.Linear(in_features=512, out_features=arg.numclass, bias=True)
        print('Model use:->> ResNet34')

    elif arg.model == 'VGG16':
        model = vgg16(pretrained=arg.PreTrain)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=arg.numclass, bias=True)
        print('Model use:->> VGG16')

    elif arg.model == 'densenet121':
        model = densenet121(pretrained=arg.PreTrain)
        model.classifier = nn.Linear(in_features=1024, out_features=arg.numclass, bias=True)
        print('Model use:->> densenet121')

    else:
        model = resnet18(pretrained=arg.PreTrain)
        model.fc = nn.Linear(in_features=512, out_features=arg.numclass, bias=True)
        print('error:-->>Model selection failed\nModel use: :->> ResNet18')

    return model

