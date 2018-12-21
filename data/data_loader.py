'''
@Description: 加载切分好的数据集
@Version: 
@Author: biofool2@gmail.com
@Date: 2018-12-21 14:49:00
@LastEditTime: 2018-12-21 16:51:50
@LastEditors: Please set LastEditors
'''

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def filepath_to_txt(txt_path, file_path):
    """将file_path中的所有文件路径写入到txt_path中"""
    f = open(txt_path, "w", encoding='utf-8')
    for root, dirs, files in os.walk(file_path):
        for sub_dir in dirs:
            img_list = glob.glob(os.path.join(file_path, sub_dir) + "\\*.png")
            lines = [img + " " + sub_dir + '\n' for img in img_list]
            f.writelines(lines)
    f.close() 

def generate_datasets_txtfile(raw_path):
    """在数据集文件同级目录下新建存放【数据集文件名+类别】的txt文本文件"""
    train_file = os.path.join(raw_path, "train.txt")
    valid_file = os.path.join(raw_path, "valid.txt")

    train_data_path = os.path.join(raw_path, "train")
    valid_data_path = os.path.join(raw_path, "valid")

    filepath_to_txt(train_file, train_data_path)
    filepath_to_txt(valid_file, valid_data_path)


class MyDataset(Dataset):
    """
    继承pytorch的Dataset类，并重写__getitem__和__len__方法
    """
    def __init__(self, txt_path, transform=None, target_transform=None):
        self.img_path = []
        self.labels = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                ip, label = line.strip().split()
                self.img_path.append(ip)
                self.labels.append(label)
        
        self.transform = transform
        self.target_transform = target_transform
    

    def __getitem__(self, index):
        ip = self.img_path[index]
        label = self.labels[index]
        # 读取图片，像素值在0-255之间
        img = Image.open(ip).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    # raw_path = r"D:\GitHub\learn_pytorch\data"
    # generate_datasets_txtfile(raw_path)
    txt_path = r"D:\GitHub\learn_pytorch\data\train.txt"

    # 数据预处理设置
    norm_mean = [0.4948052, 0.48568845, 0.44682974]
    norm_std = [0.24580306, 0.24236229, 0.2603115]
    norm_transform = transforms.Normalize(norm_mean, norm_std)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        norm_transform
    ])

    train_data = MyDataset(txt_path, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=32)
