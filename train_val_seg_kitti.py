#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to training set ground truth (.txt)', required=True)
    parser.add_argument('--filelist_val', '-v', help='Path to validation set ground truth (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%d_%s' % (args.model, args.setting, os.getpid(), time_string))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')

    print('PID:', os.getpid())

    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    print(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    sample_num = setting.sample_num
    step_val = 1000
    num_parts = setting.num_parts
    label_weights_list = setting.label_weights
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    
    #data_train, data_num_train, label_train = data_utils.load_seg(args.filelist)
    #data_val, data_num_val, label_val = data_utils.load_seg(args.filelist_val)
    data_train, img_train, data_num_train, label_train, label_hwl_train, label_xyz_train,label_ry_train, label_ry_reg_train = data_utils.load_seg_kitti(args.filelist)
    data_val, img_val, data_num_val, label_val, label_hwl_val, label_xyz_val,label_ry_val, label_ry_reg_val = data_utils.load_seg_kitti(args.filelist_val)

    # shuffle
    #data_train, data_num_train, label_train = \
    #    data_utils.grouped_shuffle([data_train, data_num_train, label_train])
    data_train, img_train, data_num_train, label_train, label_hwl_train, label_xyz_train,\
        label_ry_train, label_ry_reg_train = data_utils.grouped_shuffle([data_train, img_train,\
        data_num_train, label_train, label_hwl_train,label_xyz_train, label_ry_train,label_ry_reg_train])
        
    num_train = data_train.shape[0]
    point_num = data_train.shape[1]
    num_val = data_val.shape[0]

    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))
    batch_num = (num_train * num_epochs + batch_size - 1) // batch_size
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
    batch_num_val = math.ceil(num_val / batch_size)
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, point_num, setting.data_dim), name='pts_fts')
    labels_seg = tf.placeholder(tf.int32, shape=(None, point_num), name='labels_seg')
    labels_weights = tf.placeholder(tf.float32, shape=(None, point_num), name='labels_weights')

    labels_hwl = tf.placeholder(tf.float32, shape=(None, 3), name='labels_hwl')
    labels_xyz = tf.placeholder(tf.float32, shape=(None, 3), name='labels_xyz')
    labels_ry = tf.placeholder(tf.int32, shape=(None, 1), name='labels_ry')
    labels_ry_reg = tf.placeholder(tf.float32, shape=(None, 1), name='labels_ry_reg')
    ######################################################################

    # Set Inputs(points,features_sampled)
    features_sampled = None

    if setting.data_dim > 3:

        points, _, features = tf.split(pts_fts, [setting.data_format["pts_xyz"], setting.data_format["img_xy"], setting.data_format["extra_features"]], axis=-1, name='split_points_xy_features')

        if setting.use_extra_features:

            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')

    else:
        points = pts_fts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
    labels_weights_sampled = tf.gather_nd(labels_weights, indices=indices, name='labels_weight_sampled')
    
    # Build net
    #net = model.Net(points_augmented, features_sampled, None, None, num_parts, is_training, setting)
    net = model.Net(points_augmented, features_sampled, None, None, num_parts, is_training, setting)
    #logits, probs= net.logits, net.probs
    logits, probs, logits_hwl, logits_xyz, logits_ry, probs_ry=net.logits, net.probs, net.logits_hwl, net.logits_xyz, net.logits_ry, net.probs_ry


    # Define Loss Func
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_sampled, logits=logits, weights=labels_weights_sampled)
    _ = tf.summary.scalar('loss/train_seg', tensor=loss_op, collections=['train'])

    # HWL loss
    # 这里以MSE loss 为例
    loss_hwl_op = tf.losses.mean_squared_error(labels_hwl, logits_hwl)  # API提供的MSE loss
    # 计算RMSE来作为一种评估指标
    rmse_hwl_op = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(labels_hwl - logits_hwl), -1)))

    # XYZ loss
    # 这里以MSE loss 为例
    loss_xyz_op = tf.losses.mean_squared_error(labels_xyz, logits_xyz)
    # 计算RMSE来作为一种评估指标
    rmse_xyz_op = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(labels_xyz - logits_xyz), -1)))

    # RY loss
    # 分类以softmax_cross_entropy为例
    loss_ry_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_ry, logits=logits_ry)
    # 计算分类准确率作为评估指标
    t_1_acc_ry_op = pf.top_1_accuracy(probs_ry, labels_ry)

    # 计算平均角度偏差作为评估指标
    mdis_ry_angle_op = tf.reduce_mean(tf.sqrt(tf.square(pf.ry_getangle(labels_ry,labels_ry_reg)-
                                                        pf.ry_getangle(tf.nn.top_k(probs_ry, 1)[1], tf.zeros_like(labels_ry_reg)))))

    # 将结果可视化在Tensorboard上
    _ = tf.summary.scalar('loss/train_hwl', tensor=loss_hwl_op, collections=['train'])
    _ = tf.summary.scalar('loss/train_xyz', tensor=loss_xyz_op, collections=['train'])
    _ = tf.summary.scalar('loss/train_ry', tensor=loss_ry_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train_rmse_hwl', tensor=rmse_hwl_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train_rmse_xyz', tensor=rmse_xyz_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train_ry', tensor=t_1_acc_ry_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/mdis_ry_angle', tensor=mdis_ry_angle_op, collections=['train'])

    #for vis t1 acc
    t_1_acc_op = pf.top_1_accuracy(probs, labels_sampled)
    _ = tf.summary.scalar('t_1_acc/train_seg', tensor=t_1_acc_op, collections=['train'])
    #for vis instance acc
    t_1_acc_instance_op = pf.top_1_accuracy(probs, labels_sampled, labels_weights_sampled, 0.6)
    _ = tf.summary.scalar('t_1_acc/train_seg_instance', tensor=t_1_acc_instance_op, collections=['train'])
    #for vis other acc
    t_1_acc_others_op = pf.top_1_accuracy(probs, labels_sampled, labels_weights_sampled, 0.6,"less")
    _ = tf.summary.scalar('t_1_acc/train_seg_others', tensor=t_1_acc_others_op, collections=['train'])

    loss_val_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('loss/val_seg', tensor=loss_val_avg, collections=['val'])

    t_1_acc_val_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('t_1_acc/val_seg', tensor=t_1_acc_val_avg, collections=['val'])
    t_1_acc_val_instance_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('t_1_acc/val_seg_instance', tensor=t_1_acc_val_instance_avg, collections=['val'])
    t_1_acc_val_others_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('t_1_acc/val_seg_others', tensor=t_1_acc_val_others_avg, collections=['val'])

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()

    # 对于val，需要计算整个Val过程误差的平均值作为Val的评估指标
    loss_val_avg_hwl = tf.placeholder(tf.float32)
    loss_val_avg_xyz = tf.placeholder(tf.float32)
    loss_val_avg_ry = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('loss/val_hwl', tensor=loss_val_avg_hwl, collections=['val'])
    _ = tf.summary.scalar('loss/val_xyz', tensor=loss_val_avg_xyz, collections=['val'])
    _ = tf.summary.scalar('loss/val_ry', tensor=loss_val_avg_ry, collections=['val'])
    rmse_val_hwl_avg = tf.placeholder(tf.float32)
    rmse_val_xyz_avg = tf.placeholder(tf.float32)
    t_1_acc_val_ry_avg = tf.placeholder(tf.float32)
    mdis_val_ry_angle_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('t_1_acc/val_rmse_hwl', tensor=rmse_val_hwl_avg, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val_rmse_xyz', tensor=rmse_val_xyz_avg, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val_ry', tensor=t_1_acc_val_ry_avg, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val_mdis_ry_angle', tensor=mdis_val_ry_angle_avg, collections=['val'])

    # lr decay
    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])

    # Optimizer
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=0.9, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):

        #train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)
        train_op = optimizer.minimize(loss_op + reg_loss + loss_hwl_op + loss_xyz_op + loss_ry_op, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # backup this file, model and setting
    if not os.path.exists(os.path.join(root_folder, args.model)):
        os.makedirs(os.path.join(root_folder, args.model))
    shutil.copy(__file__, os.path.join(root_folder, os.path.basename(__file__)))
    shutil.copy(os.path.join(os.path.dirname(__file__), args.model + '.py'),
                os.path.join(root_folder, args.model + '.py'))
    shutil.copy(os.path.join(os.path.dirname(__file__), args.model.split("_")[0] + "_kitti" + '.py'),
                os.path.join(root_folder, args.model.split("_")[0] + "_kitti" + '.py'))
    shutil.copy(os.path.join(setting_path, args.setting + '.py'),
                os.path.join(root_folder, args.model, args.setting + '.py'))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    # Session Run
    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        for batch_idx in range(batch_num):
            if (batch_idx != 0 and batch_idx % step_val == 0) or batch_idx == batch_num - 1:
                ######################################################################
                # Validation
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                losses_val = []
                t_1_accs = []
                t_1_accs_instance = []
                t_1_accs_others = []

                # 声明用于计算平均值的list
                losses_val_hwl = []
                losses_val_xyz = []
                losses_val_ry = []
                rmses_hwl = []
                rmses_xyz = []
                t_1_accs_ry = []
                mdises_ry_angle = []


                for batch_val_idx in range(math.ceil(num_val / batch_size)):
                    start_idx = batch_size * batch_val_idx
                    end_idx = min(start_idx + batch_size, num_val)
                    batch_size_val = end_idx - start_idx
                    points_batch = data_val[start_idx:end_idx, ...]
                    points_num_batch = data_num_val[start_idx:end_idx, ...]
                    labels_batch = label_val[start_idx:end_idx, ...]
                    weights_batch = np.array(label_weights_list)[label_val[start_idx:end_idx, ...]]

                    # 取出每个batch的label
                    labels_hwl_batch = label_hwl_val[start_idx:end_idx, ...]
                    labels_xyz_batch = label_xyz_val[start_idx:end_idx, ...]
                    labels_ry_batch = label_ry_val[start_idx:end_idx, ...]
                    labels_ry_reg_batch = label_ry_reg_val[start_idx:end_idx, ...]

                    xforms_np, rotations_np = pf.get_xforms(batch_size_val, scaling_range=scaling_range_val)

                    sess_op_list = [loss_op, t_1_acc_op, t_1_acc_instance_op, t_1_acc_others_op]
                    sess_op_list = sess_op_list + [loss_hwl_op, loss_xyz_op, loss_ry_op, rmse_hwl_op, rmse_xyz_op,t_1_acc_ry_op, mdis_ry_angle_op]

                    sess_feed_dict = {pts_fts: points_batch,
                                     indices: pf.get_indices(batch_size_val, sample_num, points_num_batch),
                                     xforms: xforms_np,
                                     rotations: rotations_np,
                                     jitter_range: np.array([jitter_val]),
                                     labels_seg: labels_batch,
                                     labels_weights: weights_batch,
                                     is_training: False}
                    sess_feed_dict[labels_hwl] = labels_hwl_batch
                    sess_feed_dict[labels_xyz] = labels_xyz_batch
                    sess_feed_dict[labels_ry] = labels_ry_batch
                    sess_feed_dict[labels_ry_reg] = labels_ry_reg_batch

                    #loss_val, t_1_acc_val, t_1_acc_val_instance, t_1_acc_val_others = sess.run(sess_op_list,feed_dict=sess_feed_dict)
                    loss_val, t_1_acc_val, t_1_acc_val_instance, t_1_acc_val_others, loss_val_hwl, loss_val_xyz,loss_val_ry, rmse_hwl_val, rmse_xyz_val, t_1_acc_ry_val, mdis_ry_angle_val =sess.run(sess_op_list, feed_dict=sess_feed_dict)

                    #print('{}-[Val  ]-Iter: {:06d}  Loss: {:.4f} T-1 Acc: {:.4f}'.format(datetime.now(), batch_val_idx, loss_val, t_1_acc_val))
                    print('{}-[Val ]-Iter: {:06d} Loss: {:.4f} Loss_hwl: {:.4f} Loss_xyz: {:.4f} Loss_ry: {:.4f}Rmse_hwl: {:.4f} Rmse_xyz: {:.4f} T - 1Ry_Acc: {:.4f} Mdis_ry_angle: {:.4f} T - 1Acc: {:.4f}'.format(datetime.now(), batch_val_idx, loss_val, loss_val_hwl, loss_val_xyz, loss_val_ry,rmse_hwl_val, rmse_xyz_val, t_1_acc_ry_val, mdis_ry_angle_val, t_1_acc_val))

                    sys.stdout.flush()

                    losses_val.append(loss_val * batch_size_val)
                    t_1_accs.append(t_1_acc_val * batch_size_val)
                    t_1_accs_instance.append(t_1_acc_val_instance * batch_size_val)
                    t_1_accs_others.append(t_1_acc_val_others * batch_size_val)

                    # 记录每个iter的评估指标
                    losses_val_hwl.append(loss_val_hwl * batch_size_val)
                    losses_val_xyz.append(loss_val_xyz * batch_size_val)
                    losses_val_ry.append(loss_val_ry * batch_size_val)
                    rmses_hwl.append(rmse_hwl_val * batch_size_val)
                    rmses_xyz.append(rmse_xyz_val * batch_size_val)
                    t_1_accs_ry.append(t_1_acc_ry_val * batch_size_val)
                    mdises_ry_angle.append(mdis_ry_angle_val * batch_size_val)


                loss_avg = sum(losses_val)/num_val
                t_1_acc_avg = sum(t_1_accs) / num_val
                t_1_acc_instance_avg =  sum(t_1_accs_instance) / num_val
                t_1_acc_others_avg =  sum(t_1_accs_others) / num_val
                loss_avg_hwl = sum(losses_val_hwl) / num_val
                loss_avg_xyz = sum(losses_val_xyz) / num_val
                loss_avg_ry = sum(losses_val_ry) / num_val
                rmse_hwl_avg = sum(rmses_hwl) / num_val
                rmse_xyz_avg = sum(rmses_xyz) / num_val
                t_1_acc_ry_avg = sum(t_1_accs_ry) / num_val
                mdis_ry_angle_avg = sum(mdises_ry_angle) / num_val

                summaries_feed_dict = { loss_val_avg: loss_avg,
                                        t_1_acc_val_avg: t_1_acc_avg,
                                        t_1_acc_val_instance_avg: t_1_acc_instance_avg,
                                        t_1_acc_val_others_avg: t_1_acc_others_avg}
                summaries_feed_dict[loss_val_avg_hwl] = loss_avg_hwl
                summaries_feed_dict[loss_val_avg_xyz] = loss_avg_xyz
                summaries_feed_dict[loss_val_avg_ry] = loss_avg_ry
                summaries_feed_dict[rmse_val_hwl_avg] = rmse_hwl_avg
                summaries_feed_dict[rmse_val_xyz_avg] = rmse_xyz_avg
                summaries_feed_dict[t_1_acc_val_ry_avg] = t_1_acc_ry_avg
                summaries_feed_dict[mdis_val_ry_angle_avg] = mdis_ry_angle_avg

                summaries_val = sess.run(summaries_val_op,feed_dict=summaries_feed_dict)
                summary_writer.add_summary(summaries_val, batch_idx)

                #print('{}-[Val  ]-Average:      Loss: {:.4f} T-1 Acc: {:.4f}'
                #    .format(datetime.now(), loss_avg, t_1_acc_avg))
                print('{}-[Val ]-Average: Loss: {:.4f} Loss_hwl: {:.4f} Loss_xyz: {:.4f} Loss_ry: {:.4f}Rmse_hwl: {:.4f} Rmse_xyz: {:.4f} T - 1Ry_Acc: {:.4f} Mdis_ry_angle: {:.4f} T - 1Acc: {:.4f}'.format(datetime.now(), loss_avg, loss_avg_hwl, loss_avg_xyz, loss_avg_ry, rmse_hwl_avg,rmse_xyz_avg, t_1_acc_ry_avg, mdis_ry_angle_avg,t_1_acc_avg))

                sys.stdout.flush()
                ######################################################################

            ######################################################################
            # Training
            start_idx = (batch_size * batch_idx) % num_train
            end_idx = min(start_idx + batch_size, num_train)
            batch_size_train = end_idx - start_idx
            points_batch = data_train[start_idx:end_idx, ...]
            
            points_num_batch = data_num_train[start_idx:end_idx, ...]
            labels_batch = label_train[start_idx:end_idx, ...]
            weights_batch = np.array(label_weights_list)[labels_batch]

            labels_hwl_batch = label_hwl_train[start_idx:end_idx, ...]
            labels_xyz_batch = label_xyz_train[start_idx:end_idx, ...]
            labels_ry_batch = label_ry_train[start_idx:end_idx, ...]
            labels_ry_reg_batch = label_ry_reg_train[start_idx:end_idx, ...]
                
            if start_idx + batch_size_train == num_train:
                #data_train, data_num_train, label_train = \
                #    data_utils.grouped_shuffle([data_train, data_num_train, label_train])
                data_train, img_train, data_num_train, label_train, label_hwl_train, label_xyz_train, \
                    label_ry_train, label_ry_reg_train = data_utils.grouped_shuffle([data_train, img_train,data_num_train, label_train,label_hwl_train, label_xyz_train,label_ry_train, label_ry_reg_train])

            offset = int(random.gauss(0, sample_num // 8))
            offset = max(offset, -sample_num // 4)
            offset = min(offset, sample_num // 4)
            sample_num_train = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train, scaling_range=scaling_range)

            sess_op_list = [train_op, loss_op, t_1_acc_op, t_1_acc_instance_op, t_1_acc_others_op, summaries_op]
            sess_op_list = sess_op_list + [loss_hwl_op, loss_xyz_op, loss_ry_op, rmse_hwl_op, rmse_xyz_op,t_1_acc_ry_op, mdis_ry_angle_op]

            sess_feed_dict = {pts_fts: points_batch,
                             indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
                             xforms: xforms_np,
                             rotations: rotations_np,
                             jitter_range: np.array([jitter]),
                             labels_seg: labels_batch,
                             labels_weights: weights_batch,
                             is_training: True}

            # 为placeholder feed label
            sess_feed_dict[labels_hwl] = labels_hwl_batch
            sess_feed_dict[labels_xyz] = labels_xyz_batch
            sess_feed_dict[labels_ry] = labels_ry_batch
            sess_feed_dict[labels_ry_reg] = labels_ry_reg_batch

            #_, loss, t_1_acc, t_1_acc_instance, t_1_acc_others, summaries = sess.run(sess_op_list,feed_dict=sess_feed_dict)
            _, loss, t_1_acc, t_1_acc_instance, t_1_acc_others, summaries, loss_hwl, loss_xyz, loss_ry,rmse_hwl, rmse_xyz, t_1_acc_ry, mdis_ry_angle=sess.run(sess_op_list, feed_dict=sess_feed_dict)

            #print('{}-[Train]-Iter: {:06d}  Loss_seg: {:.4f} T-1 Acc: {:.4f}'
            #  .format(datetime.now(), batch_idx, loss, t_1_acc))
            print('{}-[Train]-Iter: {:06d} Loss_seg: {:.4f} Loss_hwl: {:.4f} Loss_xyz: {:.4f} Loss_ry:{:.4f} Rmse_hwl: {:.4f} Rmse_xyz: {:.4f} T - 1Ry_Acc: {:.4f} Mdis_ry_angle: {:.4f} T - 1Acc:{:.4f}'.format(datetime.now(), batch_idx, loss, loss_hwl, loss_xyz, loss_ry, rmse_hwl, rmse_xyz,t_1_acc_ry, mdis_ry_angle, t_1_acc))


            summary_writer.add_summary(summaries, batch_idx)
            sys.stdout.flush()
            
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
