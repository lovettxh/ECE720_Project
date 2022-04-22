from __init__ import corrupt
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import random
from torch.utils.data import Dataset
import numpy as np
from numpy import asarray
import cv2
random.seed(10)



def creat_dataset(rootdata):
    data_index = []
    data_list = []
    class_flag = -1
    for a,b,c in os.walk(rootdata):
        for i in range(len(c)):
            data_index.append(os.path.join(a,c[i]))
        for i in range(0,int(len(c))):
            data = os.path.join(a,c[i])+'\t'+str(class_flag)+'\n'
            data_list.append(data)
        class_flag += 1
    with open('data.txt','w') as f:
        for data in data_list:
            f.write(str(data))

class LoadData(Dataset):
    def __init__(self, txt_path):
        self.imgs_info = self.get_images(txt_path)
        self.data_tf = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.data_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)


# if __name__ == "__main__":
#     creat_dataset(rootdata)
#     dataset = LoadData("data.txt")
#     data_loader = torch.utils.data.DataLoader(dataset = dataset, shuffle = True)
#     i = 0
#     for image, label in data_loader:
#         image = image.squeeze(0)
#         image = transforms.ToPILImage()(image)
#         image = asarray(image)
#         image = corrupt(image,severity=1,corruption_number=21)
#         cv2.imwrite(str(i)+"1.jpg",image)
#         i+=1

class Common_attack():
    def __init__(self, severity):
        self.severity = severity

    def eval(self, image, corruption_number):
        image_set = []
        for i in range(image.shape[0]):
            temp = corrupt(image[i],severity=self.severity,corruption_number=corruption_number)
            image_set.append(temp)
        return np.array(image_set)
