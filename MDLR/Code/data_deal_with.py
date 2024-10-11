import os
import random
import time
from Config import load_config
args = load_config()
start_time = time.time()

data_dir = args.root_dir
output_dir = args.txt_dir
illClass = args.category

os.makedirs(output_dir,exist_ok=True)
f = open(os.path.join(output_dir, 'train_log.txt'), 'w')
f.writelines("")
f.close()
f = open(os.path.join(output_dir, 'test_log.txt'), 'w')
f.writelines("")
f.close()

train_patients_num = 0
test_patients_num = 0
pngs_count = 0
train_patients_path_list = []
test_patients_path_list = []

groupes_list = os.listdir(data_dir)
for groupes in groupes_list:
    patients_classes_path = os.path.join(data_dir,groupes)
    patients_classes_list = os.listdir(patients_classes_path)
    for patients_class in patients_classes_list:
        classes_path = os.path.join(data_dir,groupes,patients_class)
        patients_list = os.listdir(classes_path)
        for patient in patients_list:
            if groupes =='train_data':
                train_patients_num += 1
                patient_path = os.path.join(classes_path, patient)
                train_patients_path_list.append(patient_path)
            if groupes == 'test_data':
                test_patients_num += 1
                patient_path = os.path.join(classes_path, patient)
                test_patients_path_list.append(patient_path)
            pngs_count += len(os.listdir(patient_path))

random.shuffle(train_patients_path_list)
random.shuffle(test_patients_path_list)

train_mat = []
train_label = []

test_mat = []
test_label = []

data_count = 0
train_imgs_count = 0
test_imgs_count = 0
idx=0

for patient_i_path in train_patients_path_list:
    patient_class = patient_i_path.split('\\')[-2]
    png_class = 0
    if patient_class == illClass[1]:
        png_class = 1
    elif patient_class == illClass[0]:
        png_class = 0
    img_list = os.listdir(patient_i_path)

    for imgs in img_list:
        img_path = os.path.join(patient_i_path, imgs)
        data_count += 1
        train_imgs_count += 1
        f = open(os.path.join(output_dir, 'train_log.txt'), 'a')
        f.writelines((img_path,"*",str(data_count),"*", str(png_class),"\n"))
        f.close()

data_test=0
for patient_i_path in test_patients_path_list:
    patient_class = patient_i_path.split('\\')[-2]
    png_class = 0
    if patient_class == illClass[1]:
        png_class = 1
    elif patient_class == illClass[0]:
        png_class = 0

    img_list = os.listdir(patient_i_path)
    for imgs in img_list:
        img_path = os.path.join(patient_i_path, imgs)
        data_test += 1
        test_imgs_count += 1

        f = open(os.path.join(output_dir, 'test_log.txt'), 'a')
        f.writelines((img_path,"*", str(data_test),"*", str(png_class), "\n"))
        f.close()

print('TXT text saved in:{}'.format(args.txt_dir))