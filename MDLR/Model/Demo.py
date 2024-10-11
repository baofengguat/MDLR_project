import fire
import os
import sys
import torch.nn as nn
from torchvision.models import *
from Model.Check_model import Check_model

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from Model.TrainModel import train
import torch

def demo(args):

    if torch.cuda.is_available() and args.CUDA:
        print("--------CUDA has been launched-----------")

    model = Check_model(args)
    model.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,7),stride=(2, 2), padding=(3, 3), bias=False)

    print(model)

    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    if not os.path.exists(args.train_save):
        os.makedirs(args.train_save)

    train(model=model, args=args)
    print('Done!')


from Config import load_config
args = load_config()

fire.Fire(demo(args))
