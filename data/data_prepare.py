'''
@Description: 数据集准备。将CIFAR10数据集转化为图片分类别存储到本地文件夹
@Version: 
@Author: liguoying@iiotos.com
@Date: 2018-12-20 16:43:05
@LastEditTime: 2018-12-20 17:19:34
@LastEditors: Please set LastEditors
'''


import os
import numpy as np
from skimage.io import imsave


def unpickle(file):
    """
    解析数据集。解析出的数据集格式如下：
    dic = {
        b'data'         : 10000x3072维uint8类型的numpy数组，每行是张3x32x32大小的RGB图片，
                          前1024个元素是R通道，依次类推。
        b'labels'       : 10000维list，图片所属类别，取值为0-9。在batches_meta文件中保存着对应的类别名称
        b'filenames'    : 图片名称
        b'batch_label'  :
    }
    
    """
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def show_data_info(data_batch):
    """查看data_batch的基本信息"""
    print("Size : ", len(data_batch))
    print("data --> type = %s, shape = %s" %
                        type(data_batch[b'data']), data_batch[b'data'].shape)
    print("labels --> type = %s, length = %s" %
                        type(data_batch[b'labels']), len(data_batch[b'labels']))
    print("filenames --> type = %s, length = %s" %
                        type(data_batch[b'filenames']), len(data_batch[b'filenames']))
    print("batch_label --> type = %s, length = %s" %
                        type(data_batch[b'batch_label']), len(data_batch[b'batch_label']))
                        

def convert_to_png(data_batch, img_path):
    """
    将data_batch转换成png图片，并按照类别保存到相应的文件夹下。
    """
    data = data_batch[b'data']
    labels = data_batch[b'labels']
    img_names = data_batch[b'filenames']

    for i in range(10000):
        img = np.reshape(data[i], newshape=(3, 32, 32))
        img = img.transpose(1, 2, 0)

        class_path = os.path.join(img_path, str(labels[i]))
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        
        file_name = os.path.join(class_path, str(img_names[i], encoding='utf-8'))
        imsave(file_name, img)
        if i % 1000 == 0:
            print("%d images have been processed." % i)


if __name__ == '__main__':
    data_file = r"D:\GitHub\learn_pytorch\data\cifar-10-batches-py\data_batch_1"
    data_batch1 = unpickle(data_file)
    show_data_info(data_batch1)
    img_path = r"D:\GitHub\learn_pytorch\data\cifar10"
    convert_to_png(data_batch1, img_path)
