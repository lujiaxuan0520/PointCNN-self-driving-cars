#!/usr/bin/python3
'''Prepare Data for Segmentation Task.'''

#update compute train avg not all

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce

import os
import sys
import h5py
import argparse
import numpy as np
import math
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder', required=True)
    parser.add_argument('--dim', '-d', help='data dim', required=True)
    parser.add_argument('--img_scale', help='image rescale dim', default=(128,128))
    parser.add_argument('--img_padding', help='image patch padding', default=8)
    args = parser.parse_args()
    print(args)

    root = args.folder
    dim = int(args.dim)
    im_s = args.img_scale
    im_p = args.img_padding
    
    folders = [(root + 'train_data',root + 'train_img', root + 'train_label'),
               (root + 'val_data',root + 'val_img', root + 'val_label')]

    kitti_label_file = root + "frustum_label.txt"

    #read kitti label file
    label_list = [[v for v in label.split(' ')] for label in open(kitti_label_file, 'r') if label.split('_')[0] == 'fru']

    #compute avg h w l x y z (train)
    sum_list = [0.0,0.0,0.0,0.0,0.0,0.0]
    
    #get train list
    train_data_root = folders[0][0]

    train_list = [path.split(".")[0] for path in sorted(os.listdir(train_data_root))]
    train_sample_num = len(train_list)

    for label_str in label_list:

        if label_str[0].split("_")[-1] in train_list:

            sum_list[0] = sum_list[0] + float(label_str[9])
            sum_list[1] = sum_list[1] + float(label_str[10])
            sum_list[2] = sum_list[2] + float(label_str[11])
            sum_list[3] = sum_list[3] + float(label_str[12])
            sum_list[4] = sum_list[4] + float(label_str[13])
            sum_list[5] = sum_list[5] + float(label_str[14])

    avg_h = sum_list[0]/train_sample_num
    avg_w = sum_list[1]/train_sample_num
    avg_l = sum_list[2]/train_sample_num
    avg_x = sum_list[3]/train_sample_num
    avg_y = sum_list[4]/train_sample_num
    avg_z = sum_list[5]/train_sample_num

    file_avg = root + "avg.txt"

    with open(file_avg, 'w') as avg_f:

        avg_f.writelines("avg_h " + str(avg_h) + "\n")
        avg_f.writelines("avg_w " + str(avg_w) + "\n")
        avg_f.writelines("avg_l " + str(avg_l) + "\n")
        avg_f.writelines("avg_x " + str(avg_x) + "\n")
        avg_f.writelines("avg_y " + str(avg_y) + "\n")
        avg_f.writelines("avg_z " + str(avg_z) + "\n")

    print("avg_h:",avg_h)
    print("avg_w:",avg_w)
    print("avg_l:",avg_l)
    print("avg_x:",avg_x)
    print("avg_y:",avg_y)
    print("avg_z:",avg_z)

    list_ry_cls = []

    #compute offset label
    for label_str in label_list:

        #2,3 nouse

        label_str[9]  = str(float(label_str[9])  - avg_h)
        label_str[10] = str(float(label_str[10]) - avg_w)
        label_str[11] = str(float(label_str[11]) - avg_l)
        label_str[12] = str(float(label_str[12]))
        label_str[13] = str(float(label_str[13]))
        label_str[14] = str(float(label_str[14]))

        #split ry to [-12,11] -> [0,23] 24cls
        ry = float(label_str[15])/math.pi*180.0
        ry_cls = int(ry//15 + 12)
        ry_reg = float((180.0 + ry) - (ry_cls*15 + 7.5))
        label_str[15] = str(ry_cls)
        label_str[15] = str(ry_cls)
        label_str[2] = str(ry_reg)

        list_ry_cls.append(ry_cls)
        
    ulist_ry_cls,counts = np.unique(list_ry_cls,return_counts=True)
    for k,uni_ry in enumerate(ulist_ry_cls):
        print("ry_"+str(uni_ry)+":",counts[k])

    label_seg_list = np.array([])
    point_num_list = []

    for data_folder,img_folder,label_folder in folders:
        
        if not os.path.exists(data_folder):
            print(data_folder,'Source Folders Not Found')
            continue
        elif not os.path.exists(img_folder):
            print(img_folder,'Source Folders Not Found')
            continue
        elif not os.path.exists(label_folder):
            print(label_folder,'Source Folders Not Found')
            continue

        print("load data:",data_folder)

        for k,filename in enumerate(sorted(os.listdir(data_folder))):

            #print(k,filename)
            
            data_filepath = os.path.join(data_folder, filename)

            coordinates = [xyz for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == dim]
            point_num = len(coordinates)
            point_num_list.append(point_num)

            label_filepath = os.path.join(label_folder, filename[0:-3] + 'seg')
            label_seg_this = np.loadtxt(label_filepath).astype(np.int32)
            assert (len(label_seg_this) == point_num)
            label_seg_list =np.unique(np.concatenate((label_seg_list, np.unique(label_seg_this))))

    max_point_num = max(point_num_list)
    label_seg_min = min(label_seg_list)
    print("point_num_max",max(point_num_list))
    print("point_num_min",min(point_num_list))
    print("point_num_avg",reduce(lambda x, y: x + y, point_num_list) / len(point_num_list))
    print("label_segs:",label_seg_list)
        
    #h5 file batch
    batch_size = 2048
    data = np.zeros((batch_size, max_point_num, dim),dtype=np.float32)
    data_num = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)

    label_hwl = np.zeros((batch_size,3), dtype=np.float32)
    label_xyz = np.zeros((batch_size,3), dtype=np.float32)
    label_ry = np.zeros((batch_size,1), dtype=np.int32)
    label_ry_reg = np.zeros((batch_size,1), dtype=np.float32)

    data_img = np.zeros((batch_size,im_s[1],im_s[0],3), dtype=np.float32)
    
    for data_folder,img_folder, label_folder in folders:

        file_num = len(os.listdir(data_folder))
        idx_h5 = 0
        idx = 0

        save_path = '%s/%s' % (os.path.dirname(data_folder), os.path.basename(data_folder)[0:-5])
        filename_txt = '%s_files.txt' % (save_path)
        file_list = open(filename_txt, 'w')

        for k,filename in enumerate(sorted(os.listdir(data_folder))):

            data_filepath = os.path.join(data_folder, filename)
            img_filepath = os.path.join(img_folder, filename[0:-7] + ".png")
            idx_in_batch = idx % batch_size

            #point cloud and extra features
            coordinates = [[float(value) for value in xyz.split(' ')] for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == dim]

            #image
            img = Image.open(img_filepath)
                                   
            #kitti label: file_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
            label_kitti = [l for l in label_list if l[0] == 'fru_' + filename[0:-4]][0]

            #crop image
            c_2dbbox = [int(float(l_str)) for l_str in label_kitti[5:9]]
            scale_ratio = [(c_2dbbox[2]-c_2dbbox[0])/(im_s[0] - 2*im_p),(c_2dbbox[3]-c_2dbbox[1])/(im_s[1] - 2*im_p)]
            ori_p = [int(im_p*scale_ratio[0]) + 1,int(im_p*scale_ratio[1]) + 1]

            image_crop = img.crop((c_2dbbox[0] - ori_p[0], c_2dbbox[1] - ori_p[1], c_2dbbox[2] + ori_p[0], c_2dbbox[3] + ori_p[1])).resize(im_s)
            image_rgb_crop = np.array(image_crop)

            #image_crop.save("./" + filename[0:-4] + ".png")
            data_img[idx_in_batch,...] = image_rgb_crop/255

            #recompute xy and change to yx
            for c in coordinates:

                x_n = int((c[3] - (c_2dbbox[0] - ori_p[0]))/scale_ratio[0])
                y_n = int((c[4] - (c_2dbbox[1] - ori_p[1]))/scale_ratio[1])

                c[4] = x_n
                c[3] = y_n
 
            data[idx_in_batch, 0:len(coordinates), ...] = np.array(coordinates)
            data_num[idx_in_batch] = len(coordinates)

            #h w l x y z ry
            label_hwl[idx_in_batch] = [float(l_str) for l_str in label_kitti[9:12]]
            label_xyz[idx_in_batch] = [float(l_str) for l_str in label_kitti[12:15]]
            label_ry[idx_in_batch] = [int(l_str) for l_str in label_kitti[15:16]]
            label_ry_reg[idx_in_batch] = [float(l_str) for l_str in label_kitti[2:3]]

            #seg label
            label_filepath = os.path.join(label_folder, filename[0:-3] + 'seg')
            label_seg_this = np.loadtxt(label_filepath).astype(np.int32) - label_seg_min
            assert (len(coordinates) == label_seg_this.shape[0])
            label_seg[idx_in_batch, 0:len(coordinates)] = label_seg_this

            #save h5 file
            if ((idx + 1) % batch_size == 0) or idx == file_num - 1:
                item_num = idx_in_batch + 1
                filename_h5 = '%s_%d.h5' % (save_path, idx_h5)
                print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                file_list.write('./%s_%d.h5\n' % (os.path.basename(data_folder)[0:-5], idx_h5))

                file = h5py.File(filename_h5, 'w')
                file.create_dataset('data', data=data[0:item_num, ...])
                file.create_dataset('data_num', data=data_num[0:item_num, ...])
                file.create_dataset('data_img',data=data_img[0:item_num, ...])
                file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                file.create_dataset('label_hwl', data=label_hwl[0:item_num, ...])
                file.create_dataset('label_xyz', data=label_xyz[0:item_num, ...])
                file.create_dataset('label_ry', data=label_ry[0:item_num, ...])
                file.create_dataset('label_ry_reg', data=label_ry_reg[0:item_num, ...])

                file.close()

                idx_h5 = idx_h5 + 1

            idx = idx + 1

        file_list.close()

if __name__ == '__main__':
    main()
