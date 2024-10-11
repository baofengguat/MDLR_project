import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import scipy.io as scio


def img_input(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),])
    img = Image.open(img_path)
    img = transform(img)

    img.unsqueeze_(dim=0).numpy()
    return img


def feature_extract(net,conv_name,img):

    fmap_block = []
    input_block = []
    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 注册Hook
    hook = conv_name.register_forward_hook(forward_hook)
    print("--------------Extracting ---------------------",conv_name)

    with torch.no_grad():
        batch_size = 256
        img_batch = torch.split(img, batch_size, dim=0)

        k = []
        num = 0
        for imgs in img_batch:
            try:
                outputs = net(imgs.to('cuda'))
            except RuntimeError:
                outputs = net(imgs.to('cpu'))

            feature = fmap_block[num]
            num = num + 1
            k.append(feature.data.cpu().detach().numpy())
        knew = []  #
        for j in range(num):
            feature_k = k[j]
            for s in range(feature_k.shape[0]):
                feature_ks = feature_k[s, :, :, :]
                knew.append(feature_k[s])
        c = np.mean(knew, axis=0)
        feature_out = np.zeros((1, c.shape[0]), dtype="float32")
        for i in range(c.shape[0]):
            k1 = c[i, :, :]
            feature_out[0, i] = np.mean(k1[:, :])
    hook.remove()
    return feature_out


def all_feature_contact(args,input_dir,train_or_test,out_dir,train_or_test_txt):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if train_or_test == 'train_data':
        X = 'Xtrain'
        Y = 'Ytrain'
    elif train_or_test == 'test_data':
        X = 'Xtest'
        Y = 'Ytest'

    with open(train_or_test_txt,'r') as f:
        patience_list = []
        for num,line in enumerate(f):
            patience = line.strip().split('*')[0].split('\\')[-2]
            label = line.strip().split('*')[0].split('\\')[-3]
            patience_list.append(patience)
        pp = []
        for num,i in enumerate(patience_list):
            a = patience_list[num-1]
            if a != i:
                pp.append(i)
        print(pp)

    all_path_list = []
    all_name_list = []
    for i in args.category:
        path = os.path.join(input_dir,train_or_test,i)
        list = os.listdir(path)
        for j in list:
            k = os.path.join(path,j)
            all_path_list.append(k)
            all_name_list.append(j)

    index = []
    for i in pp:
        if i in all_name_list:
            index.append(all_name_list.index(i))

    patience_path = []
    label_list = []
    for j in index:
        patience_path.append(all_path_list[j])
        if all_path_list[j].split('\\')[-2] == args.category[1]:
            label_list.append(1)
        else:
            label_list.append(0)
    print(patience_path)
    print(label_list)

    train_mat = np.zeros((len(patience_path), args.feature_Number), dtype='float32')
    for num,k in enumerate(patience_path):
        mat = os.listdir(k)
        mat_path = os.path.join(k,mat[0])
        data = scio.loadmat(mat_path)['feature_map']
        train_mat[num,:] = data
    scio.savemat(os.path.join(out_dir, '%s.mat' % (train_or_test)), {X: train_mat})
    scio.savemat(os.path.join(out_dir, '%s.mat' % (train_or_test+'_label')), {Y: label_list})


def     feature_list(args,net,img):

    if args.model == 'DBB_ResNet18':
        conv1 = feature_extract(net, net.stage0.conv1.conv, img)
        # 1.0
        layer1_conv1 = feature_extract(net, net.stage0.conv1.conv, img)
        layer1_conv2 = feature_extract(net, net.stage1[0].conv2.dbb_origin.conv, img)
        layer1_conv3 = feature_extract(net, net.stage1[0].conv2.dbb_avg.conv, img)
        layer1_conv4 = feature_extract(net, net.stage1[0].conv2.dbb_1x1.conv, img)
        layer1_conv5 = feature_extract(net, net.stage1[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer1_conv6 = feature_extract(net, net.stage1[0].conv2.dbb_1x1_kxk.conv2, img)

        layer1_conv7=feature_extract(net, net.stage1[1].conv2.dbb_origin.conv, img)
        layer1_conv8 = feature_extract(net, net.stage1[1].conv2.dbb_avg.conv, img)
        layer1_conv9 = feature_extract(net, net.stage1[1].conv2.dbb_1x1.conv, img)
        layer1_conv10 = feature_extract(net, net.stage1[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer1_conv11 = feature_extract(net, net.stage1[1].conv2.dbb_1x1_kxk.conv2, img)


        # 2.0
        layer2_conv1 = feature_extract(net, net.stage2[0].shortcut.conv, img)
        layer2_conv2 = feature_extract(net, net.stage2[0].conv2.dbb_origin.conv, img)
        layer2_conv3 = feature_extract(net, net.stage2[0].conv2.dbb_avg.conv, img)
        layer2_conv4 = feature_extract(net, net.stage2[0].conv2.dbb_1x1.conv, img)
        layer2_conv5 = feature_extract(net, net.stage2[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv6 = feature_extract(net, net.stage2[0].conv2.dbb_1x1_kxk.conv2, img)

        # 2.1
        layer2_conv7 = feature_extract(net, net.stage2[1].conv2.dbb_origin.conv, img)
        layer2_conv8 = feature_extract(net, net.stage2[1].conv2.dbb_avg.conv, img)
        layer2_conv9 = feature_extract(net, net.stage2[1].conv2.dbb_1x1.conv, img)
        layer2_conv10 = feature_extract(net, net.stage2[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv11 = feature_extract(net, net.stage2[1].conv2.dbb_1x1_kxk.conv2, img)

        # 3.0
        layer3_conv1 = feature_extract(net, net.stage3[0].shortcut.conv, img)
        layer3_conv2 = feature_extract(net, net.stage3[0].conv2.dbb_origin.conv, img)
        layer3_conv3 = feature_extract(net, net.stage3[0].conv2.dbb_avg.conv, img)
        layer3_conv4 = feature_extract(net, net.stage3[0].conv2.dbb_1x1.conv, img)
        layer3_conv5 = feature_extract(net, net.stage3[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv6 = feature_extract(net, net.stage3[0].conv2.dbb_1x1_kxk.conv2, img)

        # 3.1
        layer3_conv7 = feature_extract(net, net.stage3[1].conv2.dbb_origin.conv, img)
        layer3_conv8 = feature_extract(net, net.stage3[1].conv2.dbb_avg.conv, img)
        layer3_conv9 = feature_extract(net, net.stage3[1].conv2.dbb_1x1.conv, img)
        layer3_conv10 = feature_extract(net, net.stage3[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv11 = feature_extract(net, net.stage3[1].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv19 = feature_extract(net, net.stage3[1].conv3, img)


        # 4.0
        layer4_conv1 = feature_extract(net, net.stage4[0].shortcut.conv, img)
        layer4_conv2 = feature_extract(net, net.stage4[0].conv2.dbb_origin.conv, img)
        layer4_conv3 = feature_extract(net, net.stage4[0].conv2.dbb_avg.conv, img)
        layer4_conv4 = feature_extract(net, net.stage4[0].conv2.dbb_1x1.conv, img)
        layer4_conv5 = feature_extract(net, net.stage4[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer4_conv6 = feature_extract(net, net.stage4[0].conv2.dbb_1x1_kxk.conv2, img)

        # 4.1
        layer4_conv7 = feature_extract(net, net.stage4[1].conv2.dbb_origin.conv, img)
        layer4_conv8 = feature_extract(net, net.stage4[1].conv2.dbb_avg.conv, img)
        layer4_conv9 = feature_extract(net, net.stage4[1].conv2.dbb_1x1.conv, img)
        layer4_conv10 = feature_extract(net, net.stage4[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer4_conv11 = feature_extract(net, net.stage4[1].conv2.dbb_1x1_kxk.conv2, img)

        feature_map = np.hstack((

            conv1,

            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4, layer1_conv5,
            layer1_conv6, layer1_conv7, layer1_conv8, layer1_conv9, layer1_conv10,
            layer1_conv11,

            layer2_conv1, layer2_conv2, layer2_conv3, layer2_conv4, layer2_conv5,
            layer2_conv6, layer2_conv7,  layer2_conv8, layer2_conv9,  layer2_conv10,
            layer2_conv11,

            #
            layer3_conv1, layer3_conv2, layer3_conv3, layer3_conv4, layer3_conv5,
            layer3_conv6, layer3_conv7, layer3_conv8, layer3_conv9, layer3_conv10,
            layer3_conv11,

            #
            layer4_conv1, layer4_conv2, layer4_conv3, layer4_conv4, layer4_conv5,
            layer4_conv6, layer4_conv7 ,layer4_conv8, layer4_conv9, layer4_conv10,
            layer4_conv11,

        ))

    elif args.model == 'Botnet18':
        layer1_conv1 = feature_extract(net, net[0], img)
        layer1_conv2 = feature_extract(net, net[4].net[0].shortcut[0], img)
        layer1_conv3 = feature_extract(net, net[4].net[0].net[0], img)
        layer1_conv4 = feature_extract(net, net[4].net[0].net[3].to_qkv, img)
        layer1_conv5 = feature_extract(net, net[4].net[0].net[7], img)

        layer2_conv1 = feature_extract(net, net[4].net[1].net[0], img)
        layer2_conv2 = feature_extract(net, net[4].net[1].net[3].to_qkv, img)
        layer2_conv3 = feature_extract(net, net[4].net[1].net[7], img)

        layer3_conv1 = feature_extract(net, net[4].net[2].net[0], img)
        layer3_conv2 = feature_extract(net, net[4].net[2].net[3].to_qkv, img)
        layer3_conv3 = feature_extract(net, net[4].net[2].net[7], img)

        feature_map = np.hstack((
            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4, layer1_conv5,
            layer2_conv1, layer2_conv2, layer2_conv3,
            layer3_conv1, layer3_conv2, layer3_conv3,
        ))

    elif args.model == 'ResNet18':
        conv1 = feature_extract(net, net.conv1, img)
        # 1.0
        layer1_conv1 = feature_extract(net, net.layer1[0].conv1, img)
        layer1_conv2 = feature_extract(net, net.layer1[0].conv2, img)

        layer1_conv3 = feature_extract(net, net.layer1[1].conv1, img)
        layer1_conv4 = feature_extract(net, net.layer1[1].conv2, img)

        layer2_conv1 = feature_extract(net, net.layer2[0].conv1, img)
        layer2_conv2 = feature_extract(net, net.layer2[0].conv2, img)

        layer2_conv3 = feature_extract(net, net.layer2[1].conv1, img)
        layer2_conv4 = feature_extract(net, net.layer2[1].conv2, img)

        layer3_conv1 = feature_extract(net, net.layer3[0].conv1, img)
        layer3_conv2 = feature_extract(net, net.layer3[0].conv2, img)

        layer3_conv3 = feature_extract(net, net.layer3[1].conv1, img)
        layer3_conv4 = feature_extract(net, net.layer3[1].conv2, img)

        layer4_conv1 = feature_extract(net, net.layer4[0].conv1, img)
        layer4_conv2 = feature_extract(net, net.layer4[0].conv2, img)

        layer4_conv3 = feature_extract(net, net.layer4[1].conv1, img)
        layer4_conv4 = feature_extract(net, net.layer4[1].conv2, img)

        feature_map = np.hstack((
            #
            conv1,
            #
            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4,
            layer2_conv1, layer2_conv2, layer2_conv3, layer2_conv4,
            layer3_conv1, layer3_conv2, layer3_conv3, layer3_conv4,
            layer4_conv1, layer4_conv2, layer4_conv3, layer4_conv4,

        ))

    elif args.model == 'ResNet34':
        conv1 = feature_extract(net, net.conv1, img)
        # 1.0
        layer1_conv1 = feature_extract(net, net.layer1[0].conv1, img)
        layer1_conv2 = feature_extract(net, net.layer1[0].conv2, img)
        layer1_conv3 = feature_extract(net, net.layer1[1].conv1, img)
        layer1_conv4 = feature_extract(net, net.layer1[1].conv2, img)
        layer1_conv5 = feature_extract(net, net.layer1[2].conv1, img)
        layer1_conv6 = feature_extract(net, net.layer1[2].conv2, img)

        layer2_conv1 = feature_extract(net, net.layer2[0].conv1, img)
        layer2_conv2 = feature_extract(net, net.layer2[0].conv2, img)
        layer2_conv3 = feature_extract(net, net.layer2[1].conv1, img)
        layer2_conv4 = feature_extract(net, net.layer2[1].conv2, img)
        layer2_conv5 = feature_extract(net, net.layer2[2].conv1, img)
        layer2_conv6 = feature_extract(net, net.layer2[2].conv2, img)
        layer2_conv7 = feature_extract(net, net.layer2[3].conv1, img)
        layer2_conv8 = feature_extract(net, net.layer2[3].conv2, img)

        layer3_conv1 = feature_extract(net, net.layer3[0].conv1, img)
        layer3_conv2 = feature_extract(net, net.layer3[0].conv2, img)
        layer3_conv3 = feature_extract(net, net.layer3[1].conv1, img)
        layer3_conv4 = feature_extract(net, net.layer3[1].conv2, img)
        layer3_conv5 = feature_extract(net, net.layer3[2].conv1, img)
        layer3_conv6 = feature_extract(net, net.layer3[2].conv2, img)
        layer3_conv7 = feature_extract(net, net.layer3[3].conv1, img)
        layer3_conv8 = feature_extract(net, net.layer3[3].conv2, img)
        layer3_conv9 = feature_extract(net, net.layer3[4].conv1, img)
        layer3_conv10 = feature_extract(net, net.layer3[4].conv2, img)
        layer3_conv11 = feature_extract(net, net.layer3[5].conv1, img)
        layer3_conv12 = feature_extract(net, net.layer3[5].conv2, img)

        layer4_conv1 = feature_extract(net, net.layer4[0].conv1, img)
        layer4_conv2 = feature_extract(net, net.layer4[0].conv2, img)
        layer4_conv3 = feature_extract(net, net.layer4[1].conv1, img)
        layer4_conv4 = feature_extract(net, net.layer4[1].conv2, img)
        layer4_conv5 = feature_extract(net, net.layer4[2].conv1, img)
        layer4_conv6 = feature_extract(net, net.layer4[2].conv2, img)

        feature_map = np.hstack((
            #
            conv1,
            #
            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4,
            layer1_conv5, layer1_conv6,

            layer2_conv1, layer2_conv2, layer2_conv3, layer2_conv4,
            layer2_conv5, layer2_conv6, layer2_conv7, layer2_conv8,

            layer3_conv1, layer3_conv2, layer3_conv3, layer3_conv4,
            layer3_conv5, layer3_conv6, layer3_conv7, layer3_conv8,
            layer3_conv9, layer3_conv10, layer3_conv11, layer3_conv12,

            layer4_conv1, layer4_conv2, layer4_conv3, layer4_conv4,
            layer4_conv5, layer4_conv6,

        ))

    elif args.model == 'Botnet34':
        layer1_conv1 = feature_extract(net, net[0], img)
        layer1_conv2 = feature_extract(net, net[4][0].conv1, img)
        layer1_conv3 = feature_extract(net, net[4][0].conv2, img)

        layer1_conv4 = feature_extract(net, net[4][1].conv1, img)
        layer1_conv5 = feature_extract(net, net[4][1].conv2, img)

        layer1_conv6 = feature_extract(net, net[4][2].conv1, img)
        layer1_conv7 = feature_extract(net, net[4][2].conv2, img)

        layer1_conv8 = feature_extract(net, net[5].net[0].shortcut[0], img)
        layer1_conv9 = feature_extract(net, net[5].net[0].net[0], img)
        layer1_conv10 = feature_extract(net, net[5].net[0].net[3].to_qkv, img)
        layer1_conv11 = feature_extract(net, net[5].net[0].net[7], img)

        layer2_conv1 = feature_extract(net, net[5].net[1].net[0], img)
        layer2_conv2 = feature_extract(net, net[5].net[1].net[3].to_qkv, img)
        layer2_conv3 = feature_extract(net, net[5].net[1].net[7], img)

        layer3_conv1 = feature_extract(net, net[5].net[2].net[0], img)
        layer3_conv2 = feature_extract(net, net[5].net[2].net[3].to_qkv, img)
        layer3_conv3 = feature_extract(net, net[5].net[2].net[7], img)

        feature_map = np.hstack((
            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4, layer1_conv5,
            layer1_conv6, layer1_conv7, layer1_conv8, layer1_conv9, layer1_conv10,
            layer1_conv11,
            layer2_conv1, layer2_conv2, layer2_conv3,
            layer3_conv1, layer3_conv2, layer3_conv3,
        ))

    elif args.model == 'VGG16':
        # 1.0
        layer1_conv1 = feature_extract(net, net.features[0], img)
        layer1_conv2 = feature_extract(net, net.features[2], img)
        layer1_conv3 = feature_extract(net, net.features[5], img)
        layer1_conv4 = feature_extract(net, net.features[7], img)
        layer1_conv5 = feature_extract(net, net.features[10], img)
        layer1_conv6 = feature_extract(net, net.features[12], img)
        layer1_conv7 = feature_extract(net, net.features[14], img)
        layer1_conv8 = feature_extract(net, net.features[17], img)
        layer1_conv9 = feature_extract(net, net.features[19], img)
        layer1_conv10 = feature_extract(net, net.features[21], img)
        layer1_conv11 = feature_extract(net, net.features[24], img)
        layer1_conv12 = feature_extract(net, net.features[26], img)
        layer1_conv13 = feature_extract(net, net.features[28], img)

        feature_map = np.hstack((
            layer1_conv1, layer1_conv2, layer1_conv3, layer1_conv4, layer1_conv5,
            layer1_conv6, layer1_conv7, layer1_conv8, layer1_conv9, layer1_conv10,
            layer1_conv11,layer1_conv12,layer1_conv13,

        ))

    elif args.model == 'densenet121':
        conv1 = feature_extract(net, net.features.conv0, img)
        # #1.0
        # layer1_conv1 = feature_extract(net, net.features.denseblock1.denselayer1.conv1, img)
        # layer1_conv3 = feature_extract(net, net.features.denseblock1.denselayer1.conv2, img)
        # layer1_conv5 = feature_extract(net, net.features.denseblock1.denselayer2.conv1, img)
        # layer1_conv6 = feature_extract(net, net.features.denseblock1.denselayer2.conv2, img)
        # layer1_conv7 = feature_extract(net, net.features.denseblock1.denselayer3.conv1, img)
        # layer1_conv8 = feature_extract(net, net.features.denseblock1.denselayer3.conv2, img)
        layer1_conv9 = feature_extract(net, net.features.denseblock1.denselayer4.conv1, img)
        layer1_conv11 = feature_extract(net, net.features.denseblock1.denselayer4.conv2, img)
        layer1_conv13 = feature_extract(net, net.features.denseblock1.denselayer5.conv1, img)
        layer1_conv14 = feature_extract(net, net.features.denseblock1.denselayer5.conv2, img)
        layer1_conv15 = feature_extract(net, net.features.denseblock1.denselayer6.conv1, img)
        layer1_conv16 = feature_extract(net, net.features.denseblock1.denselayer6.conv2, img)
        layer1_conv17 = feature_extract(net, net.features.transition1.conv, img)

        # 2.0
        # layer2_conv1 = feature_extract(net, net.features.denseblock2.denselayer1.conv1, img)
        # layer2_conv3 = feature_extract(net, net.features.denseblock2.denselayer1.conv2, img)
        # layer2_conv5 = feature_extract(net, net.features.denseblock2.denselayer2.conv1, img)
        # layer2_conv6 = feature_extract(net, net.features.denseblock2.denselayer2.conv2, img)
        # layer2_conv7 = feature_extract(net, net.features.denseblock2.denselayer3.conv1, img)
        # layer2_conv8 = feature_extract(net, net.features.denseblock2.denselayer3.conv2, img)
        layer2_conv9 = feature_extract(net, net.features.denseblock2.denselayer4.conv1, img)
        layer2_conv11 = feature_extract(net, net.features.denseblock2.denselayer4.conv2, img)
        layer2_conv13 = feature_extract(net, net.features.denseblock2.denselayer5.conv1, img)
        layer2_conv14 = feature_extract(net, net.features.denseblock2.denselayer5.conv2, img)
        layer2_conv15 = feature_extract(net, net.features.denseblock2.denselayer6.conv1, img)
        layer2_conv16 = feature_extract(net, net.features.denseblock2.denselayer6.conv2, img)
        layer2_conv33 = feature_extract(net, net.features.transition2.conv, img)

        # 3.0
        # layer3_conv1 = feature_extract(net, net.features.denseblock3.denselayer1.conv1, img)
        # layer3_conv3 = feature_extract(net, net.features.denseblock3.denselayer1.conv2, img)
        # layer3_conv5 = feature_extract(net, net.features.denseblock3.denselayer2.conv1, img)
        # layer3_conv6 = feature_extract(net, net.features.denseblock3.denselayer2.conv2, img)
        # layer3_conv7 = feature_extract(net, net.features.denseblock3.denselayer3.conv1, img)
        # layer3_conv8 = feature_extract(net, net.features.denseblock3.denselayer3.conv2, img)
        # layer3_conv9 = feature_extract(net, net.features.denseblock3.denselayer4.conv1, img)
        # layer3_conv11 = feature_extract(net, net.features.denseblock4.denselayer3.conv2, img)
        layer3_conv13 = feature_extract(net, net.features.denseblock3.denselayer5.conv1, img)
        layer3_conv14 = feature_extract(net, net.features.denseblock3.denselayer5.conv2, img)
        layer3_conv15 = feature_extract(net, net.features.denseblock3.denselayer6.conv1, img)
        layer3_conv16 = feature_extract(net, net.features.denseblock3.denselayer6.conv2, img)
        layer3_conv17 = feature_extract(net, net.features.denseblock3.denselayer7.conv1, img)
        layer3_conv18 = feature_extract(net, net.features.denseblock3.denselayer7.conv2, img)
        layer3_conv20 = feature_extract(net, net.features.denseblock3.denselayer8.conv1, img)
        layer3_conv22 = feature_extract(net, net.features.denseblock3.denselayer8.conv2, img)
        layer3_conv23 = feature_extract(net, net.features.denseblock3.denselayer9.conv1, img)
        layer3_conv24 = feature_extract(net, net.features.denseblock3.denselayer9.conv2, img)
        layer3_conv25 = feature_extract(net, net.features.denseblock3.denselayer10.conv1, img)
        layer3_conv26 = feature_extract(net, net.features.denseblock3.denselayer10.conv2, img)
        layer3_conv27 = feature_extract(net, net.features.denseblock3.denselayer11.conv1, img)
        layer3_conv29 = feature_extract(net, net.features.denseblock3.denselayer11.conv2, img)
        layer3_conv31 = feature_extract(net, net.features.denseblock3.denselayer12.conv1, img)
        layer3_conv32 = feature_extract(net, net.features.denseblock3.denselayer12.conv2, img)
        layer3_conv33 = feature_extract(net, net.features.transition3.conv, img)

        # # 4.0
        layer4_conv1 = feature_extract(net, net.features.denseblock4.denselayer1.conv1, img)
        layer4_conv3 = feature_extract(net, net.features.denseblock4.denselayer1.conv2, img)
        layer4_conv5 = feature_extract(net, net.features.denseblock4.denselayer2.conv1, img)
        layer4_conv6 = feature_extract(net, net.features.denseblock4.denselayer2.conv2, img)
        layer4_conv7 = feature_extract(net, net.features.denseblock4.denselayer3.conv1, img)
        layer4_conv8 = feature_extract(net, net.features.denseblock4.denselayer3.conv2, img)
        layer4_conv9 = feature_extract(net, net.features.denseblock4.denselayer4.conv1, img)
        layer4_conv11 = feature_extract(net, net.features.denseblock4.denselayer4.conv2, img)
        layer4_conv13 = feature_extract(net, net.features.denseblock4.denselayer5.conv1, img)
        layer4_conv14 = feature_extract(net, net.features.denseblock4.denselayer5.conv2, img)
        layer4_conv15 = feature_extract(net, net.features.denseblock4.denselayer6.conv1, img)
        layer4_conv16 = feature_extract(net, net.features.denseblock4.denselayer6.conv2, img)
        layer4_conv17 = feature_extract(net, net.features.denseblock4.denselayer7.conv1, img)
        layer4_conv18 = feature_extract(net, net.features.denseblock4.denselayer7.conv2, img)
        layer4_conv20 = feature_extract(net, net.features.denseblock4.denselayer8.conv1, img)
        layer4_conv22 = feature_extract(net, net.features.denseblock4.denselayer8.conv2, img)
        layer4_conv23 = feature_extract(net, net.features.denseblock4.denselayer9.conv1, img)
        layer4_conv24 = feature_extract(net, net.features.denseblock4.denselayer9.conv2, img)
        layer4_conv25 = feature_extract(net, net.features.denseblock4.denselayer10.conv1, img)
        layer4_conv26 = feature_extract(net, net.features.denseblock4.denselayer10.conv2, img)
        layer4_conv27 = feature_extract(net, net.features.denseblock4.denselayer11.conv1, img)
        layer4_conv29 = feature_extract(net, net.features.denseblock4.denselayer11.conv2, img)

        layer4_conv30 = feature_extract(net, net.features.denseblock4.denselayer12.conv1, img)
        layer4_conv31 = feature_extract(net, net.features.denseblock4.denselayer12.conv2, img)
        layer4_conv32 = feature_extract(net, net.features.denseblock4.denselayer13.conv1, img)
        layer4_conv33 = feature_extract(net, net.features.denseblock4.denselayer13.conv2, img)
        layer4_conv34 = feature_extract(net, net.features.denseblock4.denselayer14.conv1, img)
        layer4_conv35 = feature_extract(net, net.features.denseblock4.denselayer14.conv2, img)
        layer4_conv36 = feature_extract(net, net.features.denseblock4.denselayer15.conv1, img)
        layer4_conv37 = feature_extract(net, net.features.denseblock4.denselayer15.conv2, img)
        layer4_conv38 = feature_extract(net, net.features.denseblock4.denselayer16.conv1, img)
        layer4_conv39 = feature_extract(net, net.features.denseblock4.denselayer16.conv2, img)

        feature_map = np.hstack((
            #
            conv1,
            #
            # layer1_conv1, layer1_conv3, layer1_conv5, layer1_conv6, layer1_conv7,
            # layer1_conv8,
            layer1_conv9, layer1_conv11,
            layer1_conv13, layer1_conv14, layer1_conv15, layer1_conv16,
            layer1_conv17,
            #
            # layer2_conv1, layer2_conv3,layer2_conv5, layer2_conv6, layer2_conv7,
            # layer2_conv8,
            layer2_conv9, layer2_conv11,
            layer2_conv13, layer2_conv14, layer2_conv15, layer2_conv16,
            layer2_conv33,

            #
            # layer3_conv1,  layer3_conv3,layer3_conv5, layer3_conv6, layer3_conv7,
            # layer3_conv8, layer3_conv9, layer3_conv11,
            layer3_conv13, layer3_conv14, layer3_conv15, layer3_conv16,
            layer3_conv17, layer3_conv18,
            layer3_conv20, layer3_conv22,
            layer3_conv23, layer3_conv24, layer3_conv25,
            layer3_conv26, layer3_conv27,
            layer3_conv29, layer3_conv31, layer3_conv32,
            layer3_conv33,
            #
            layer4_conv1, layer4_conv3, layer4_conv5, layer4_conv6, layer4_conv7,
            layer4_conv8, layer4_conv9,
            layer4_conv11, layer4_conv13, layer4_conv14, layer4_conv15, layer4_conv16,
            layer4_conv17, layer4_conv18,
            layer4_conv20, layer4_conv22, layer4_conv23, layer4_conv24, layer4_conv25, layer4_conv26,
            layer4_conv27, layer4_conv29,
            layer4_conv30, layer4_conv31, layer4_conv32, layer4_conv33, layer4_conv34, layer4_conv35,
            layer4_conv36, layer4_conv37, layer4_conv38, layer4_conv39

        ))

    elif args.model == 'DBB_ResNet50':
        conv1 = feature_extract(net, net.stage0.conv1.conv, img)
        # 1.0
        layer1_conv1 = feature_extract(net, net.stage1[0].shortcut.conv, img)
        # layer1_conv2 = feature_extract(net, net.stage1[0].conv1, img)
        layer1_conv3 = feature_extract(net, net.stage1[0].conv1.conv, img)
        # layer1_conv4 = feature_extract(net, net.stage1[0].conv2, img)
        layer1_conv5 = feature_extract(net, net.stage1[0].conv2.dbb_origin.conv, img)
        layer1_conv6 = feature_extract(net, net.stage1[0].conv2.dbb_avg.conv, img)
        layer1_conv7 = feature_extract(net, net.stage1[0].conv2.dbb_1x1.conv, img)
        layer1_conv8 = feature_extract(net, net.stage1[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer1_conv9 = feature_extract(net, net.stage1[0].conv2.dbb_1x1_kxk.conv2, img)
        # layer1_conv10 = feature_extract(net, net.stage1[0].conv3, img)
        layer1_conv11 = feature_extract(net, net.stage1[0].conv3.conv, img)

        # 1.1
        # layer1_conv12 = feature_extract(net, net.stage1[1].conv1, img)
        layer1_conv13 = feature_extract(net, net.stage1[1].conv1.conv, img)
        layer1_conv14 = feature_extract(net, net.stage1[1].conv2.dbb_origin.conv, img)
        layer1_conv15 = feature_extract(net, net.stage1[1].conv2.dbb_avg.conv, img)
        layer1_conv16 = feature_extract(net, net.stage1[1].conv2.dbb_1x1.conv, img)
        layer1_conv17 = feature_extract(net, net.stage1[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer1_conv18 = feature_extract(net, net.stage1[1].conv2.dbb_1x1_kxk.conv2, img)
        # layer1_conv19 = feature_extract(net, net.stage1[1].conv3, img)
        layer1_conv20 = feature_extract(net, net.stage1[1].conv3.conv, img)

        # 1.2
        # layer1_conv21 = feature_extract(net, net.stage1[2].conv1, img)
        layer1_conv22 = feature_extract(net, net.stage1[2].conv1.conv, img)
        layer1_conv23 = feature_extract(net, net.stage1[2].conv2.dbb_origin.conv, img)
        layer1_conv24 = feature_extract(net, net.stage1[2].conv2.dbb_avg.conv, img)
        layer1_conv25 = feature_extract(net, net.stage1[2].conv2.dbb_1x1.conv, img)
        layer1_conv26 = feature_extract(net, net.stage1[2].conv2.dbb_1x1_kxk.idconv1, img)
        layer1_conv27 = feature_extract(net, net.stage1[2].conv2.dbb_1x1_kxk.conv2, img)
        # layer1_conv28 = feature_extract(net, net.stage1[2].conv3, img)
        layer1_conv29 = feature_extract(net, net.stage1[2].conv3.conv, img)

        # 2.0
        layer2_conv1 = feature_extract(net, net.stage2[0].shortcut.conv, img)
        # layer2_conv2 = feature_extract(net, net.stage2[0].conv1, img)
        layer2_conv3 = feature_extract(net, net.stage2[0].conv1.conv, img)
        # layer2_conv4 = feature_extract(net, net.stage2[0].conv2, img)
        layer2_conv5 = feature_extract(net, net.stage2[0].conv2.dbb_origin.conv, img)
        layer2_conv6 = feature_extract(net, net.stage2[0].conv2.dbb_avg.conv, img)
        layer2_conv7 = feature_extract(net, net.stage2[0].conv2.dbb_1x1.conv, img)
        layer2_conv8 = feature_extract(net, net.stage2[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv9 = feature_extract(net, net.stage2[0].conv2.dbb_1x1_kxk.conv2, img)
        # layer2_conv10 = feature_extract(net, net.stage2[0].conv3, img)
        layer2_conv11 = feature_extract(net, net.stage2[0].conv3.conv, img)

        # 2.1
        # layer2_conv12 = feature_extract(net, net.stage2[1].conv1, img)
        layer2_conv13 = feature_extract(net, net.stage2[1].conv1.conv, img)
        layer2_conv14 = feature_extract(net, net.stage2[1].conv2.dbb_origin.conv, img)
        layer2_conv15 = feature_extract(net, net.stage2[1].conv2.dbb_avg.conv, img)
        layer2_conv16 = feature_extract(net, net.stage2[1].conv2.dbb_1x1.conv, img)
        layer2_conv17 = feature_extract(net, net.stage2[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv18 = feature_extract(net, net.stage2[1].conv2.dbb_1x1_kxk.conv2, img)
        # layer2_conv19 = feature_extract(net, net.stage2[1].conv3, img)
        layer2_conv20 = feature_extract(net, net.stage2[1].conv3.conv, img)

        # 2.2
        # layer2_conv21 = feature_extract(net, net.stage2[2].conv1, img)
        layer2_conv22 = feature_extract(net, net.stage2[2].conv1.conv, img)
        layer2_conv23 = feature_extract(net, net.stage2[2].conv2.dbb_origin.conv, img)
        layer2_conv24 = feature_extract(net, net.stage2[2].conv2.dbb_avg.conv, img)
        layer2_conv25 = feature_extract(net, net.stage2[2].conv2.dbb_1x1.conv, img)
        layer2_conv26 = feature_extract(net, net.stage2[2].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv27 = feature_extract(net, net.stage2[2].conv2.dbb_1x1_kxk.conv2, img)
        # layer2_conv28 = feature_extract(net, net.stage2[2].conv3, img)
        layer2_conv29 = feature_extract(net, net.stage2[2].conv3.conv, img)

        # 2.3
        # layer2_conv30 = feature_extract(net, net.stage2[3].conv1, img)
        layer2_conv31 = feature_extract(net, net.stage2[3].conv1.conv, img)
        layer2_conv32 = feature_extract(net, net.stage2[3].conv2.dbb_origin.conv, img)
        layer2_conv33 = feature_extract(net, net.stage2[3].conv2.dbb_avg.conv, img)
        layer2_conv34 = feature_extract(net, net.stage2[3].conv2.dbb_1x1.conv, img)
        layer2_conv35 = feature_extract(net, net.stage2[3].conv2.dbb_1x1_kxk.idconv1, img)
        layer2_conv36 = feature_extract(net, net.stage2[3].conv2.dbb_1x1_kxk.conv2, img)
        # layer2_conv37 = feature_extract(net, net.stage2[3].conv3, img)
        layer2_conv38 = feature_extract(net, net.stage2[3].conv3.conv, img)

        # 3.0
        layer3_conv1 = feature_extract(net, net.stage3[0].shortcut.conv, img)
        # layer3_conv2 = feature_extract(net, net.stage3[0].conv1, img)
        layer3_conv3 = feature_extract(net, net.stage3[0].conv1.conv, img)
        # layer3_conv4 = feature_extract(net, net.stage3[0].conv2, img)
        layer3_conv5 = feature_extract(net, net.stage3[0].conv2.dbb_origin.conv, img)
        layer3_conv6 = feature_extract(net, net.stage3[0].conv2.dbb_avg.conv, img)
        layer3_conv7 = feature_extract(net, net.stage3[0].conv2.dbb_1x1.conv, img)
        layer3_conv8 = feature_extract(net, net.stage3[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv9 = feature_extract(net, net.stage3[0].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv10 = feature_extract(net, net.stage3[0].conv3, img)
        layer3_conv11 = feature_extract(net, net.stage3[0].conv3.conv, img)

        # 3.1
        #  layer3_conv12 = feature_extract(net, net.stage3[1].conv1, img)
        layer3_conv13 = feature_extract(net, net.stage3[1].conv1.conv, img)
        layer3_conv14 = feature_extract(net, net.stage3[1].conv2.dbb_origin.conv, img)
        layer3_conv15 = feature_extract(net, net.stage3[1].conv2.dbb_avg.conv, img)
        layer3_conv16 = feature_extract(net, net.stage3[1].conv2.dbb_1x1.conv, img)
        layer3_conv17 = feature_extract(net, net.stage3[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv18 = feature_extract(net, net.stage3[1].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv19 = feature_extract(net, net.stage3[1].conv3, img)
        layer3_conv20 = feature_extract(net, net.stage3[1].conv3.conv, img)

        # 2.2
        # layer3_conv21 = feature_extract(net, net.stage3[2].conv1, img)
        layer3_conv22 = feature_extract(net, net.stage3[2].conv1.conv, img)
        layer3_conv23 = feature_extract(net, net.stage3[2].conv2.dbb_origin.conv, img)
        layer3_conv24 = feature_extract(net, net.stage3[2].conv2.dbb_avg.conv, img)
        layer3_conv25 = feature_extract(net, net.stage3[2].conv2.dbb_1x1.conv, img)
        layer3_conv26 = feature_extract(net, net.stage3[2].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv27 = feature_extract(net, net.stage3[2].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv28 = feature_extract(net, net.stage3[2].conv3, img)
        layer3_conv29 = feature_extract(net, net.stage3[2].conv3.conv, img)

        # 3.3
        # layer3_conv30 = feature_extract(net, net.stage3[3].conv1, img)
        layer3_conv31 = feature_extract(net, net.stage3[3].conv1.conv, img)
        layer3_conv32 = feature_extract(net, net.stage3[3].conv2.dbb_origin.conv, img)
        layer3_conv33 = feature_extract(net, net.stage3[3].conv2.dbb_avg.conv, img)
        layer3_conv34 = feature_extract(net, net.stage3[3].conv2.dbb_1x1.conv, img)
        layer3_conv35 = feature_extract(net, net.stage3[3].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv36 = feature_extract(net, net.stage3[3].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv37 = feature_extract(net, net.stage3[3].conv3, img)
        layer3_conv38 = feature_extract(net, net.stage3[3].conv3.conv, img)

        # 3.4
        # layer3_conv39 = feature_extract(net, net.stage3[4].conv1, img)
        layer3_conv40 = feature_extract(net, net.stage3[4].conv1.conv, img)
        layer3_conv41 = feature_extract(net, net.stage3[4].conv2.dbb_origin.conv, img)
        layer3_conv42 = feature_extract(net, net.stage3[4].conv2.dbb_avg.conv, img)
        layer3_conv43 = feature_extract(net, net.stage3[4].conv2.dbb_1x1.conv, img)
        layer3_conv44 = feature_extract(net, net.stage3[4].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv45 = feature_extract(net, net.stage3[4].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv46 = feature_extract(net, net.stage3[4].conv3, img)
        layer3_conv47 = feature_extract(net, net.stage3[4].conv3.conv, img)

        # 3.5
        # layer3_conv48 = feature_extract(net, net.stage3[5].conv1, img)
        layer3_conv49 = feature_extract(net, net.stage3[5].conv1.conv, img)
        layer3_conv50 = feature_extract(net, net.stage3[5].conv2.dbb_origin.conv, img)
        layer3_conv51 = feature_extract(net, net.stage3[5].conv2.dbb_avg.conv, img)
        layer3_conv52 = feature_extract(net, net.stage3[5].conv2.dbb_1x1.conv, img)
        layer3_conv53 = feature_extract(net, net.stage3[5].conv2.dbb_1x1_kxk.idconv1, img)
        layer3_conv54 = feature_extract(net, net.stage3[5].conv2.dbb_1x1_kxk.conv2, img)
        # layer3_conv55 = feature_extract(net, net.stage3[5].conv3, img)
        layer3_conv56 = feature_extract(net, net.stage3[5].conv3.conv, img)

        # 4.0
        layer4_conv1 = feature_extract(net, net.stage4[0].shortcut.conv, img)
        # layer4_conv2 = feature_extract(net, net.stage4[0].conv1, img)
        layer4_conv3 = feature_extract(net, net.stage4[0].conv1.conv, img)
        # layer4_conv4 = feature_extract(net, net.stage4[0].conv2, img)
        layer4_conv5 = feature_extract(net, net.stage4[0].conv2.dbb_origin.conv, img)
        layer4_conv6 = feature_extract(net, net.stage4[0].conv2.dbb_avg.conv, img)
        layer4_conv7 = feature_extract(net, net.stage4[0].conv2.dbb_1x1.conv, img)
        layer4_conv8 = feature_extract(net, net.stage4[0].conv2.dbb_1x1_kxk.idconv1, img)
        layer4_conv9 = feature_extract(net, net.stage4[0].conv2.dbb_1x1_kxk.conv2, img)
        # layer4_conv10 = feature_extract(net, net.stage4[0].conv3, img)
        layer4_conv11 = feature_extract(net, net.stage4[0].conv3.conv, img)

        # 4.1
        # layer4_conv12 = feature_extract(net, net.stage4[1].conv1, img)
        layer4_conv13 = feature_extract(net, net.stage4[1].conv1.conv, img)
        layer4_conv14 = feature_extract(net, net.stage4[1].conv2.dbb_origin.conv, img)
        layer4_conv15 = feature_extract(net, net.stage4[1].conv2.dbb_avg.conv, img)
        layer4_conv16 = feature_extract(net, net.stage4[1].conv2.dbb_1x1.conv, img)
        layer4_conv17 = feature_extract(net, net.stage4[1].conv2.dbb_1x1_kxk.idconv1, img)
        layer4_conv18 = feature_extract(net, net.stage4[1].conv2.dbb_1x1_kxk.conv2, img)
        # layer4_conv19 = feature_extract(net, net.stage4[1].conv3, img)
        layer4_conv20 = feature_extract(net, net.stage4[1].conv3.conv, img)

        # 4.2
        # layer4_conv21 = feature_extract(net, net.stage4[2].conv1, img)
        layer4_conv22 = feature_extract(net, net.stage4[2].conv1.conv, img)
        layer4_conv23 = feature_extract(net, net.stage4[2].conv2.dbb_origin.conv, img)
        layer4_conv24 = feature_extract(net, net.stage4[2].conv2.dbb_avg.conv, img)
        layer4_conv25 = feature_extract(net, net.stage4[2].conv2.dbb_1x1.conv, img)
        layer4_conv26 = feature_extract(net, net.stage4[2].conv2.dbb_1x1_kxk.idconv1, img)
        layer4_conv27 = feature_extract(net, net.stage4[2].conv2.dbb_1x1_kxk.conv2, img)
        # layer4_conv28 = feature_extract(net, net.stage4[2].conv3, img)
        layer4_conv29 = feature_extract(net, net.stage4[2].conv3.conv, img)

        feature_map = np.hstack((
            #
            conv1,
            #
            layer1_conv1, layer1_conv3, layer1_conv5, layer1_conv6, layer1_conv7,
            layer1_conv8, layer1_conv9,
            layer1_conv11, layer1_conv13, layer1_conv14, layer1_conv15, layer1_conv16,
            layer1_conv17, layer1_conv18,
            layer1_conv20, layer1_conv22, layer1_conv23, layer1_conv24, layer1_conv25, layer1_conv26,
            layer1_conv27, layer1_conv29,
            #
            layer2_conv1, layer2_conv3, layer2_conv5, layer2_conv6, layer2_conv7,
            layer2_conv8, layer2_conv9,
            layer2_conv11, layer2_conv13, layer2_conv14, layer2_conv15, layer2_conv16,
            layer2_conv17, layer2_conv18,
            layer2_conv20, layer2_conv22, layer2_conv23, layer2_conv24, layer2_conv25, layer2_conv26,
            layer2_conv27, layer2_conv29,
            layer2_conv31, layer2_conv32, layer2_conv33, layer2_conv34, layer2_conv35, layer2_conv36,
            layer2_conv38,
            #
            layer3_conv1, layer3_conv3, layer3_conv5, layer3_conv6, layer3_conv7,
            layer3_conv8, layer3_conv9,
            layer3_conv11, layer3_conv13, layer3_conv14, layer3_conv15, layer3_conv16,
            layer3_conv17, layer3_conv18,
            layer3_conv20, layer3_conv22, layer3_conv23, layer3_conv24, layer3_conv25, layer3_conv26, layer3_conv27,
            layer3_conv29,
            layer3_conv31, layer3_conv32, layer3_conv33, layer3_conv34, layer3_conv35, layer3_conv36,
            layer3_conv38,
            layer3_conv40, layer3_conv41, layer3_conv42, layer3_conv43, layer3_conv44, layer3_conv45,
            layer3_conv47, layer3_conv49,
            layer3_conv50, layer3_conv51, layer3_conv52, layer3_conv53, layer3_conv54, layer3_conv56,
            #
            layer4_conv1, layer4_conv3, layer4_conv5, layer4_conv6, layer4_conv7,
            layer4_conv8, layer4_conv9,
            layer4_conv11, layer4_conv13, layer4_conv14, layer4_conv15, layer4_conv16, layer4_conv17,
            layer4_conv18,
            layer4_conv20, layer4_conv22, layer4_conv23, layer4_conv24, layer4_conv25, layer4_conv26,
            layer4_conv27, layer4_conv29,

        ))

    print("a img feature map all shape:{}".format(feature_map.shape))
    args.feature_Number = feature_map.shape[1]
    return feature_map





