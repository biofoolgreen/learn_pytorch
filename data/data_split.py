'''
@Description: 切分数据集
@Version: 
@Author: biofool2@gmail.com
@Date: 2018-12-20 17:38:33
@LastEditTime: 2018-12-21 14:36:43
@LastEditors: Please set LastEditors
'''

import os
import numpy as np
import random
import glob
import shutil


def gen_dirs(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

def gen_train_test_datasets(rawdata_path, test_size=0.15, valid_size=0.15, random_seed=1024):
    assert test_size+valid_size < 0.5, "测试集和验证集数据总数的比例不超过50%。"
    root_path = os.path.split(rawdata_path)[0]
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')
    valid_path = os.path.join(root_path, 'valid')

    gen_dirs(train_path)
    gen_dirs(test_path)
    gen_dirs(valid_path)

    random.seed(random_seed)
    for root, dirs, files in os.walk(rawdata_path):
        # 遍历源数据文件夹
        for sub_dir in dirs:
            img_list = glob.glob(os.path.join(root, sub_dir) + '\\*.png')
            num_img = len(img_list)
            num_test = int(test_size * num_img)
            num_valid = int(valid_size * num_img)
            # 随机打乱文件列表
            random.shuffle(img_list)

            test_imgs = img_list[:num_test]
            valid_imgs = img_list[num_test:(num_test+num_valid)]
            train_imgs = img_list[(num_test+num_valid):]

            for img in img_list:
                if img in test_imgs:
                    out_dir = os.path.join(test_path, sub_dir)
                elif img in valid_imgs:
                    out_dir = os.path.join(valid_path, sub_dir)
                else:
                    out_dir = os.path.join(train_path, sub_dir)
                # print("out_dir : ", out_dir)
                
                gen_dirs(out_dir)
                img_name = os.path.split(img)[-1]
                out_file = os.path.join(out_dir, img_name)
                
                shutil.copy(img, out_file)
            print("目录 [%s] 下的文件移动完毕。" % sub_dir)


if __name__ == '__main__':
    raw_path = r"D:\GitHub\learn_pytorch\data\cifar10"
    gen_train_test_datasets(raw_path)
