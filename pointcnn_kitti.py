from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf


def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)

    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')
    
    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training, with_bn=False)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training,  with_bn=False)
        #fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        #fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN:
    def __init__(self, points, features, images, image_xy, num_class, is_training, setting, task):

        print("points:",points)
        print("features:",features)

        xconv_params = setting.xconv_params
        print("xconv_params:",xconv_params)

        fc_params = setting.fc_params
        print("fc_params:", fc_params)

        # 从配置文件中获取后续分支的参数
        extra_branch_xconv_params = setting.extra_branch_xconv_params
        extra_branch_fc_params = setting.extra_branch_fc_params
        mask_sample_num = setting.mask_sample_num
        
        with_X_transformation = setting.with_X_transformation
        sorting_method = setting.sorting_method

        N = tf.shape(points)[0]
        point_num = tf.shape(points)[1]

        if setting.with_fps:
            from sampling import tf_sampling

        self.layer_pts = [points]
        
        if features is None:
            self.layer_fts = [features]
        else:
            C_fts = xconv_params[0][-1] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K, D, P, C = layer_param

            fts = self.layer_fts[-1]

            # get k centroid points
            pts = self.layer_pts[-1]
            
            if P == -1:
                qrs = self.layer_pts[-1]
            else:
                if setting.with_fps:
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    fps_indices_g = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, fps_indices_g, name= tag + 'qrs') # (N, P, 3)
                else:
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)

            self.layer_pts.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1][-1]
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)

            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                              depth_multiplier, sorting_method,False)
            self.layer_fts.append(fts_xconv)


        if task == 'segmentation':
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K, D, pts_layer_idx, qrs_layer_idx = layer_param

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                _, _, P, C = xconv_params[qrs_layer_idx]

                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

        self.fc_layers = [self.layer_fts[-1]]
        for layer_idx, layer_param in enumerate(fc_params):
            channel_num, drop_rate = layer_param
            fc = pf.dense(self.fc_layers[-1], channel_num, 'fc{:d}'.format(layer_idx), is_training)
            fc_drop = tf.layers.dropout(fc, drop_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)

        if task == 'classification':
            fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
            self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)

        self.logits = pf.dense(self.fc_layers[-1], num_class, 'logits', is_training, with_bn=False, activation=None)
        self.probs = tf.nn.softmax(self.logits, name='probs')

        # 从分割结果获取Mask点云(车辆点云)
        probs_others = tf.slice(self.probs, [0, 0, 0], [N, point_num, 1])
        probs_instance = tf.slice(self.probs, [0, 0, 1], [N, point_num, 1])
        hold_sub_mask = tf.subtract(probs_instance, probs_others, name="hold_sub_mask")
        indices_mask = tf.py_func(pf.instance_choice, [hold_sub_mask, mask_sample_num], tf.int32)
        indices_mask.set_shape([hold_sub_mask.get_shape()[0], mask_sample_num, 2])
        points_mask = tf.gather_nd(points, indices_mask, name='points_mask')

        # 计算mask点云重心
        if setting.with_xyz_res:
            points_mask_mean = tf.reduce_mean(points_mask, axis=-2, keep_dims=True,name='points_mask_mean')
            points_mask_mean_sq = tf.squeeze(points_mask_mean, name='points_mask_mean_sq')
            points_mask_centroid = tf.subtract(points_mask, points_mask_mean, name='points_mask_centroid')
            eb_pts = points_mask_centroid
        else:
            eb_pts = points_mask
        # 获取mask点云特征
        if features is None:
            features_mask_hd = None
        else:
            C_fts = extra_branch_xconv_params[0][-1] // 2
            features_mask = tf.gather_nd(features, indices_mask, name='features_mask')
            mask_features_hd = pf.dense(features_mask, C_fts, 'mask_features_hd', is_training)
        eb_fts = mask_features_hd

        # Bbox分支(尺寸与位置分支)
        self.bbox_layer_fts = [eb_fts]
        self.bbox_layer_pts = [eb_pts]
        for layer_idx, layer_param in enumerate(extra_branch_xconv_params):
            tag = 'bbox_branch_xconv_' + str(layer_idx + 1) + '_'
            K, D, P, C = layer_param
            pts = self.bbox_layer_pts[-1]
            fts = self.bbox_layer_fts[-1]
            # get centroid points
            if P == -1:
                qrs = eb_pts
            else:
                if setting.eb_with_fps:
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    fps_indices_g = tf.concat([batch_indices, tf.expand_dims(fps_indices, -1)], axis=-1)
                    qrs = tf.gather_nd(pts, fps_indices_g, name=tag + 'qrs')  # (N, P, 3)
                else:
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            self.bbox_layer_pts.append(qrs)
            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1][-1]
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training,
                              with_X_transformation, depth_multiplier, sorting_method, False)
            self.bbox_layer_fts.append(fts_xconv)

        # HWL 预测分支 以用maxpooling特征进行预测为例
        self.hwl_fc_layers = [tf.reduce_max(self.bbox_layer_fts[-1], axis=1, name='hwl_fts_max')]
        for layer_idx, layer_param in enumerate(extra_branch_fc_params):
            channel_num, drop_rate = layer_param
            fc = pf.dense(self.hwl_fc_layers[-1], channel_num, 'hwl_fc{:d}'.format(layer_idx),
                          is_training, with_bn=setting.hwl_branch_setting["with_bn"])
            fc_drop = tf.layers.dropout(fc, drop_rate, training=is_training,
                                        name='hwl_fc{:d}_drop'.format(layer_idx))
            self.hwl_fc_layers.append(fc_drop)
        self.logits_hwl = pf.dense(self.hwl_fc_layers[-1], 3, 'logits_hwl', is_training, with_bn=False,
                                   activation=None)

        # XYZ 预测分支 以用maxpooling特征进行预测为例
        self.xyz_fc_layers = [tf.reduce_max(self.bbox_layer_fts[-1], axis=1, name='xyz_fts_max')]
        for layer_idx, layer_param in enumerate(extra_branch_fc_params):
            channel_num, drop_rate = layer_param
            fc = pf.dense(self.xyz_fc_layers[-1], channel_num, 'xyz_fc{:d}'.format(layer_idx),
                          is_training, with_bn=setting.xyz_branch_setting["with_bn"])
            fc_drop = tf.layers.dropout(fc, drop_rate, training=is_training,
                                        name='xyz_fc{:d}_drop'.format(layer_idx))
            self.xyz_fc_layers.append(fc_drop)
        if setting.with_xyz_res:
            logits_xyz_res = pf.dense(self.xyz_fc_layers[-1], 3, 'logits_xyz_res', is_training,
                                      with_bn=False, activation=None)
            self.logits_xyz = tf.add(points_mask_mean_sq, logits_xyz_res, name='logits_xyz')
        else:
            self.logits_xyz = pf.dense(self.xyz_fc_layers[-1], 3, 'logits_xyz_res', is_training,
                                       with_bn=False, activation=None)

        # Ry分支(旋转角分支)
        self.ry_layer_fts = [eb_fts]
        self.ry_layer_pts = [eb_pts]
        for layer_idx, layer_param in enumerate(extra_branch_xconv_params):
            tag = 'ry_branch_xconv_' + str(layer_idx + 1) + '_'
            K, D, P, C = layer_param
            fts = self.ry_layer_fts[-1]
            # get centroid points
            pts = self.ry_layer_pts[-1]
            if P == -1:
                qrs = eb_pts
            else:
                if setting.eb_with_fps:
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    qrs = tf_sampling.gather_point(pts, fps_indices)  # (N,P,3)
                else:
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            self.ry_layer_pts.append(qrs)
            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1][-1]
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training,
                              with_X_transformation, depth_multiplier, sorting_method, False)
            self.ry_layer_fts.append(fts_xconv)

        # Ry 分类预测分支 以用maxpooling特征进行预测为例
        self.ry_fc_layers = [tf.reduce_max(self.ry_layer_fts[-1], axis=1, name='ry_fts_max')]
        for layer_idx, layer_param in enumerate(extra_branch_fc_params):
            channel_num, drop_rate = layer_param
            fc = pf.dense(self.ry_fc_layers[-1], channel_num, 'ry_fc{:d}'.format(layer_idx),
                          is_training, with_bn=setting.ry_branch_setting["with_bn"])
            fc_drop = tf.layers.dropout(fc, drop_rate, training=is_training,
                                        name='ry_fc{:d}_drop'.format(layer_idx))
            self.ry_fc_layers.append(fc_drop)
        self.logits_ry = pf.dense(self.ry_fc_layers[-1], 24, 'logits_ry', is_training,
                                  with_bn=False, activation=None)
        self.probs_ry = tf.nn.softmax(self.logits_ry, name='probs_ry')