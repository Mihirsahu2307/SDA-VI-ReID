import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = './SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
class SYSUData_DA(data.Dataset):
    def __init__(self, data_dir, num_ids=20, use_test = False, transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = './SYSU-MM01/'
        # Load training images (path) and labels
        
        if use_test:
            train_color_image_og = np.load(data_dir + 'train_large_rgb_resized_img.npy')
            train_color_label_og = np.load(data_dir + 'train_large_rgb_resized_label.npy')

            train_thermal_image_og = np.load(data_dir + 'train_large_ir_resized_img.npy')
            train_thermal_label_og = np.load(data_dir + 'train_large_ir_resized_label.npy')
        else:
            train_color_image_og = np.load(data_dir + 'train_rgb_resized_img.npy')
            train_color_label_og = np.load(data_dir + 'train_rgb_resized_label.npy')

            train_thermal_image_og = np.load(data_dir + 'train_ir_resized_img.npy')
            train_thermal_label_og = np.load(data_dir + 'train_ir_resized_label.npy')
    
        
        train_color_image, train_thermal_image, train_color_label, train_thermal_label  = [], [], [], []
        
        # Generate 20 random indices from unique elements of train_color_label_og
        len_tot = len(np.unique(train_color_label_og))
        
        if use_test:
            indices = range(len_tot)
        else:
            indices = np.random.choice(len_tot, num_ids, replace=False)
        
        # print(indices)
        indices = set(indices)
        # indices = set(range(num_ids))
        
        label_mapping = {label: i for i, label in enumerate(indices)}
        
        for i in range(len(train_color_label_og)):
            if train_color_label_og[i] in indices:
                train_color_image.append(train_color_image_og[i])
                train_color_label.append(label_mapping[train_color_label_og[i]])
                
        for i in range(len(train_thermal_label_og)):
            if train_thermal_label_og[i] in indices:
                train_thermal_image.append(train_thermal_image_og[i])
                train_thermal_label.append(label_mapping[train_thermal_label_og[i]])
                
        # print("Loop Done!!\n")
        
        # BGR to RGB
        self.train_color_image   = np.array(train_color_image)
        self.train_thermal_image = np.array(train_thermal_image)
        self.train_color_label = np.array(train_color_label)
        self.train_thermal_label = np.array(train_thermal_label)
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        print("SYSU Loading Done!!\n")
        # print("Total Color images chosen: " + str(len(train_color_image)))
        # print("Total Thermal images chosen: " + str(len(train_thermal_image)))
        

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = './RegDB/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
    
    
# Add Test set images for training too:
class RegDBData_DA(data.Dataset):
    def __init__(self, data_dir, trial, num_ids = 20, use_test = False, transform=None, colorIndex = None, thermalIndex = None):
        """ 
        num_ids is the number of train ids to be chosen for training
        Note: For RegDB, since each id has 10 training images per modality, slicing the array is straightforward
        """
        
        # Load training images (path) and labels
        data_dir = './RegDB/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        # Note: train_thermal_label is always 0, 1, 2, .. 205 in that order, regardless of true PID as it doesn't matter
        
        train_color_image = []
        
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
                    
        if use_test:
            test_color_list   = data_dir + 'idx/test_visible_{}'.format(trial)+ '.txt'
            test_thermal_list = data_dir + 'idx/test_thermal_{}'.format(trial)+ '.txt'
            test_color_img_file, test_color_label = load_data(test_color_list, offset=206)
            test_thermal_img_file, test_thermal_label = load_data(test_thermal_list, offset=206)
            for i in range(len(test_color_img_file)):
                img = Image.open(data_dir+ color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                train_color_image.append(pix_array)
            for i in range(len(test_thermal_img_file)):
                img = Image.open(data_dir+ thermal_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                train_thermal_image.append(pix_array)
                
            train_color_label.extend(test_color_label)
            train_thermal_label.extend(test_thermal_label)
            
        
        train_color_image = np.array(train_color_image) 
        train_thermal_image = np.array(train_thermal_image)
        
        if not use_test:
            self.train_color_image = train_color_image[:num_ids * 10]
            self.train_color_label = train_color_label[:num_ids * 10]
            
            self.train_thermal_image = train_thermal_image[:num_ids * 10]
            self.train_thermal_label = train_thermal_label[:num_ids * 10]
        else:
            self.train_color_image = train_color_image
            self.train_color_label = train_color_label
            
            self.train_thermal_image = train_thermal_image
            self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex        

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path, offset = 0):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) + offset for s in data_file_list]
        
    return file_image, file_label
