import os
import torchvision.transforms as transforms
from PIL import Image
from Model.FeaturePack import *
import torch
import numpy as np
from Model.Check_model import Check_model
import scipy.io as scio
import time
import torch.nn as nn

from Config import load_config
args = load_config()

net = Check_model(args)
net.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),stride=(2, 2), padding=(3, 3), bias=False)
dir = args.root_dir
out_dir = os.path.join(args.train_save,args.model,'feature_extract')
os.makedirs(out_dir,exist_ok=True)
PATH = os.path.join(args.train_save,args.model, args.model+'_model.pth')

checkpoint = torch.load(PATH,map_location="cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(checkpoint)
net.eval()

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i in os.listdir(dir):
    data_dir = os.path.join(dir, i)
    for j in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, j)
        for n,k in enumerate(os.listdir(class_dir)):
            mat_file = os.path.join(out_dir,i,j,k)
            if not os.path.exists(mat_file):
                os.makedirs(mat_file)

            mat_file = os.path.join(mat_file,'%s.mat'%(k))
            if not os.path.exists(mat_file):
                patience_dir = os.path.join(class_dir, k)
                number = len(os.listdir(patience_dir))
                start_time = time.time()
                feature_out = np.zeros((number, 1, 224, 224), dtype="float32")
                for num, o in enumerate(os.listdir(patience_dir)):
                    img_path = os.path.join(patience_dir, o)
                    img = img_input(img_path)
                    feature_out[num, :, :, :] = img
                bitch_img = torch.from_numpy(feature_out)
                print(bitch_img.shape)

                feature_map = feature_list(args,net, bitch_img)
                scio.savemat(mat_file, {'feature_map': feature_map})
                over_time = time.time()
                print('{} feature_map Feature extraction complete cost time: {}s'.format(k,(over_time-start_time)))

                all = len(os.listdir(class_dir))
                print("当前 {} | {}:Completed  {}  /  总计  {}".format(i, j, n+1, all))

            else:
                print('%s.mat feature_map Already exists'%k)

input_dir = os.path.join(args.train_save,args.model,'feature_extract')
train_txt = os.path.join(args.txt_dir,'train_log.txt')
test_txt = os.path.join(args.txt_dir, 'test_log.txt')
out_dir1 = os.path.join(args.train_save,args.model,"feature_extract_concation")
all_feature_contact(args,input_dir,'train_data',out_dir1,train_txt)
all_feature_contact(args,input_dir, 'test_data', out_dir1, test_txt)

