#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-f', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--data_folder', '-d', help='Path to *.pts directory', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--sample_num', help='Point sample num', type=int, default=1024)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)

    args = parser.parse_args()
    print(args)

    model = importlib.import_module(args.model)
    sys.path.append(os.path.dirname(args.setting))
    setting = importlib.import_module(os.path.basename(args.setting))

    sample_num = setting.sample_num
    num_parts = setting.num_parts

    output_folder = os.path.abspath(os.path.join(args.data_folder, "..")) + '/pred_' + str(args.repeat_num)

    output_folder_seg = output_folder + '/seg/'
    output_folder_bbox = output_folder + '/bbox/'

    # check the path
    if not os.path.exists(output_folder_seg):
        print(output_folder_seg, "Not Exists! Create", output_folder_seg)
        os.makedirs(output_folder_seg)
    if not os.path.exists(output_folder_bbox):
        print(output_folder_bbox, "Not Exists! Create", output_folder_bbox)
        os.makedirs(output_folder_bbox)

    input_filelist = []
    output_seg_filelist = []
    output_bbox_filelist = []

    for filename in sorted(os.listdir(args.data_folder)):
        input_filelist.append(os.path.join(args.data_folder, filename))
        output_seg_filelist.append(os.path.join(output_folder_seg, filename[0:-3] + 'seg'))
        output_bbox_filelist.append(os.path.join(output_folder_bbox, filename[0:-3] + 'txt'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))

    data, data_num, _label_seg, _label_hwl, _label_xyz, _label_ry = data_utils.load_seg_kitti(args.filelist)
    
    batch_num = data.shape[0]
    max_point_num = data.shape[1]
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='pts_fts')

#    if setting.with_img_conv:
#        img_fts = tf.placeholder(tf.float32, shape=(batch_size, img_h, img_w, img_c), name='img_fts')
    #######################################################################

    features_augmented = None

    if setting.data_dim > 3:

        points, _, features = tf.split(pts_fts, [setting.data_format["pts_xyz"], setting.data_format["img_xy"], setting.data_format["extra_features"]], axis=-1, name='split_points_xy_features')

        if setting.use_extra_features:

            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')

            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, setting.data_dim-6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                features_augmented = features_sampled
    else:
        points = pts_fts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')



    net = model.Net(points_sampled, features_augmented, None, None, num_parts, is_training, setting)

    
    probs_op, logits_hwl_op, logits_xyz_op, probs_ry_op= net.probs, net.logits_hwl, net.logits_xyz, net.probs_ry


    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        for batch_idx in range(batch_num):

            points_batch = data[[batch_idx] * batch_size, ...]
            point_num = data_num[batch_idx]

           
            tile_num = math.ceil((sample_num * batch_size) / point_num)
            indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
            np.random.shuffle(indices_shuffle)
            indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
            indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

            sess_op_list = [probs_op]

            sess_feed_dict = {pts_fts: points_batch,
                              indices: indices_batch,
                              is_training: False}
            sess_op_list = sess_op_list + [logits_hwl_op, logits_xyz_op, probs_ry_op]


            #sess run
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            probs, logits_hwl, logits_xyz, probs_ry = sess.run(sess_op_list,feed_dict=sess_feed_dict)



            #output seg probs
            probs_2d = np.reshape(probs, (sample_num * batch_size, -1)) 
            predictions = [(-1, 0.0, [])] * point_num

            for idx in range(sample_num * batch_size):
                point_idx = indices_shuffle[idx]
                point_probs = probs_2d[idx, :]
                prob = np.amax(point_probs)
                seg_idx = np.argmax(point_probs)
                if prob > predictions[point_idx][1]:
                    predictions[point_idx] = [seg_idx, prob, point_probs]

            with open(output_seg_filelist[batch_idx], 'w') as file_seg:
                for seg_idx, prob, probs in predictions:
                    file_seg.write(str(int(seg_idx)) + "\n")

            print('{}-[Testing]-Iter: {:06d} \nseg  saved to {}'.format(datetime.now(), batch_idx, output_seg_filelist[batch_idx]))

             #output hwl xyz ry
            if setting.with_extra_branch:

                predictions_ry = [-1, 0.0, []]

                for b_idx in range(batch_size):

                    ry_probs = probs_ry[b_idx, :]
                    prob_ry =  np.amax(ry_probs)
                    ry_idx = np.argmax(ry_probs)

                    if prob_ry > predictions_ry[1]:
                        predictions_ry = [ry_idx, prob_ry, ry_probs]


                #avg
                hwl = np.average(logits_hwl,axis=0)
                xyz = np.average(logits_xyz,axis=0)


            with open(output_bbox_filelist[batch_idx], 'w') as file_bbox:

                file_bbox.write("hwl " + str(hwl[0]) + " " + str(hwl[1]) + " " + str(hwl[2]) + "\n")
                file_bbox.write("xyz " + str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + "\n")
                file_bbox.write("ry " + str(int(predictions_ry[0])) + "\n")

                print('bbox saved to {}'.format(output_bbox_filelist[batch_idx]))
            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))

if __name__ == '__main__':
    main()
