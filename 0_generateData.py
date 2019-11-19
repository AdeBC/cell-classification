
import cv2
import os
from matplotlib import pyplot as plt
from random import sample, randint, seed
import numpy as np
'''
需求：
    找到图中的所有轮廓，并按照轮廓的外接正方形切割成小图片
    保存为src/ori_image/cell_*.jpg

---
现有问题：
    二值化结果不够理想，无法将距离近的细胞边缘分割
    
    解决方法：再调参数

保存算法需要改进
    尝试：从左上角开始，固定矩形的大小，而后再进行sample将其分为train和test数据集

读取文件时，使用ls命令将文件名写入到txt里面，然后再用for循环遍历形成np数组

'''


def findContours(image): 
    '''
        返回所有细胞轮廓
    '''
    
    image = cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)#图像降噪
    blur_img = cv2.medianBlur(image,5)#图像滤波

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image =  cv2.adaptiveThreshold(image, 255, \
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, 289, -3)#局部阈值方法，所选参数比ADAPTIVE_THRESH_MEAN_C方法较优
    
    cv2.imwrite('二值化.png', image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('\tfound contours')
    c = cv2.drawContours(image, contours, -1,(0,255,0), 1)
    return contours


def generateTrueCell(image, contours):
    '''
        使用40*40的矩形框进行裁剪，并以列表类型返回所有细胞小图
        后续：删除错误数据、人工给予标记
    '''

    images_list= []

    for i,contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        center_x = int(x+ w/2)
        center_y = int(y+ h/2) 

        if min( w,h ) > 10 and max( w,h ) < 50:         
            new_image= image[center_y-20:center_y+20,center_x-20:center_x+20]
            
            if new_image.shape[0] == new_image.shape[1]:
                images_list.append(new_image)

    return images_list             


def generateFakeCell(image, k):
    images_list= []
    seed= 5
    for i in range(k):
        center_x= randint(20, image.shape[0]-20)
        center_y= randint(20, image.shape[1]-20)
        new_image = image[center_y-20:center_y+20, center_x-20:center_x+20]
        images_list.append(new_image)
    
    return images_list


def keepStandardPic(images_list):
    '''
        使用for循环进行遍历，依次显示每张图片并进行删除标记
        删除存在多个细胞粘连的图片
    '''

    for index, pic in enumerate(images_list):
        p= cv2.resize(pic,(600,600),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image'+str(index), p)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        remove = input('remove this pic? y/[n] :')
        
        if remove=='y':
            images_list.pop(index)
    
    return images_list


data             = {}
image            = cv2.imread('cell.png')
contours         = findContours(image)

trueCell_keep    = generateTrueCell(image, contours)
print('filtering true cell!')
# trueCell_keep  = keepStandardPic(trueCell)
data['trueCell'] = trueCell_keep 
data['trueLabel']= ['1']*len(trueCell_keep)

fakeCell         = generateFakeCell(image, 20)
print('filtering fake cell!')
fakeCell_keep    = keepStandardPic(fakeCell)*20
data['fakeCell'] = fakeCell_keep
data['fakeLabel']= ['0']*len(fakeCell_keep)

directory= '/home/u201713020/Desktop/cell_classification/src/'
data_merge = {'cell': data['trueCell'] + data['fakeCell'],\
    'label': data['trueLabel'] + data['fakeLabel']
    }

if not os.path.isdir(directory):
    os.makedirs(directory)

for index,pic in enumerate(data_merge['cell']):
    dir= directory + 'data/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    cv2.imwrite(dir+'cell_'+str(index)+'.png', pic)

f=open(directory+'label.txt', 'w')
f.write( ','.join(data_merge['label']) )
f.close()
