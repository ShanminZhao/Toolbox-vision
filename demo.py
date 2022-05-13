# -*- coding: utf-8 -*-
import cv2
import os
import json
import numpy as np

from utils.common import mkdir
from utils import image_processing

global img
global point1, point2
global g_rect
 

def on_mouse(event, x, y, flags, param):
    global img, point1, point2,g_rect
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        # print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
 
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        # print("2-EVENT_FLAG_LBUTTON")
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)
 
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        # print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        if point1!=point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect=[min_x,min_y,width,height]
            cut_img = img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('ROI', cut_img)
 
def get_image_roi(rgb_image):
    '''
    获得用户ROI区域的rect=[x,y,w,h]
    :param rgb_image:
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    global img
    img=bgr_image
    cv2.namedWindow('image')
    while True:
        cv2.setMouseCallback('image', on_mouse)
        # cv2.startWindowThread()  # 加在这个位置
        cv2.imshow('image', img)
        key=cv2.waitKey(0)
        if key==13 or key==32:#按空格和回车键退出
            break
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect
 
def select_user_roi(image_path):
    '''
    由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
    :param image_path:
    :return:
    '''
    orig_image = image_processing.read_image(image_path)
    orig_shape = np.shape(orig_image)
    resize_image = image_processing.resize_image(orig_image, resize_height=None,resize_width=None)
    re_shape = np.shape(resize_image)
    g_rect=get_image_roi(resize_image)
    orgi_rect = image_processing.scale_rect(g_rect, re_shape,orig_shape)
    roi_image=image_processing.get_rect_image(orig_image,orgi_rect)
    # image_processing.cv_show_image("RECT",roi_image)
    # image_processing.show_image_rect("image",orig_image,orgi_rect)
    return orgi_rect, roi_image
 
 
if __name__ == '__main__':
    images = os.listdir('./data')
    mkdir('roi_data')
    mkdir('roi_label')
    for index, img_name in enumerate(images):
        new_name = str(index+1) + '.bmp'
        image_path=os.path.join('data', img_name)
        orig_image = image_processing.read_image(image_path) #RGB
        # rect=get_image_roi(orig_image)
        rect, roi_image=select_user_roi(image_path)
        cv2.imwrite(os.path.join('roi_data', new_name), roi_image)
        
        start_x, start_y = rect[0], rect[1]
        label_path = os.path.join('label', img_name+'.json')
        with open(label_path, 'r', encoding='utf-8') as file_in:
            json_dict = json.load(file_in) 
            json_dict[ "ImagePath"] = new_name
            blocks = json_dict["Labels"]
            for block in blocks:
                for point in block['Points']:
                    point['X'] -= start_x
                    point['Y'] -= start_y
            file_out = open(os.path.join('roi_label', new_name+'.json'), 'w')
            json.dump(json_dict, file_out, indent = 4)
            file_out.close()