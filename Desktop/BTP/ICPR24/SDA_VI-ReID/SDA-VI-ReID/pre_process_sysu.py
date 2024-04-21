import numpy as np
from PIL import Image
import pdb
import os

# Modified https://github.com/mangye16/Cross-Modal-Re-ID-baseline/blob/master/pre_process_sysu.py

data_path = './SYSU-MM01/'

rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
file_path_test  = os.path.join(data_path,'exp/test_id.txt')

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    
with open(file_path_test, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_test = ["%04d" % x for x in ids]
    
# All VI-ReID baselines combine train and val split
id_train.extend(id_val) 

# In case, you want to use train+val+test for source domain in Stage-2
# id_train.extend(id_test)

files_rgb = []
files_ir = []

print("Begin\n\n")
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
            
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
             
# test_files_rgb = []
# test_files_ir = []
# for id in sorted(id_test):
#     for cam in rgb_cameras:
#         img_dir = os.path.join(data_path,cam,id)
#         if os.path.isdir(img_dir):
#             new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
#             test_files_rgb.extend(new_files)
            
#     for cam in ir_cameras:
#         img_dir = os.path.join(data_path,cam,id)
#         if os.path.isdir(img_dir):
#             new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
#             test_files_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
    
print(len(pid_container))
pid2label = {pid:label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288
def read_imgs(train_image):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array) 
        
        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)
       
# rgb imges
train_img, train_label = read_imgs(files_rgb)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
train_img, train_label = read_imgs(files_ir)
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)

# # rgb imges
# train_img, train_label = read_imgs(files_rgb)
# np.save(data_path + 'train_large_rgb_resized_img.npy', train_img)
# np.save(data_path + 'train_large_rgb_resized_label.npy', train_label)

# # ir imges
# train_img, train_label = read_imgs(files_ir)
# np.save(data_path + 'train_large_ir_resized_img.npy', train_img)
# np.save(data_path + 'train_large_ir_resized_label.npy', train_label)

print("Done creating .npy files")