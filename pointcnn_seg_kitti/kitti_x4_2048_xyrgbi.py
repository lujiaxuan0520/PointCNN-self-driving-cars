#!/usr/bin/python3
import math

#switch

num_parts = 2

sample_num = 2048

batch_size = 12

num_epochs = 1024

label_weights = []

for c in range(num_parts):

	if c == 0:

		label_weights.append(0.4)

	else:

		label_weights.append(1.0)

learning_rate_base = 0.005
decay_steps = 20000
decay_rate = 0.8
learning_rate_min = 1e-6

weight_decay = 0.0
#weight_decay = 1e-5

#jitter = 0.001
jitter = 0.0

jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0, 0, 0, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 4

with_img_conv = True
# C Kernel Strides Padding
img_conv_params = [(16 * x,(9,9),(1,1),(4,4)),
                   (32 * x,(5,5),(1,1),(2,2)),
                   (64 * x,(3,3),(1,1),(1,1)),
                   (64 * x,(3,3),(1,1),(1,1))]

#branch config
with_extra_branch = True

#mask coordinate
mask_sample_num = 1024
with_xyz_res = True
mask_centroid_type = "z" # xyz/z

#mask feature
with_fruxyz_features = True
with_seg_features = False

xyz_branch_setting = {"with_bn":False,"type":"points_pred","loss":"pts_pred_wmse"} # type:[maxpooling/mean_pred/points_pred] loss:[mse,w_mse,pts_pred_wmse]
hwl_branch_setting = {"with_bn":True,"type":"mean_pred","loss":"w_mse"} #type:[maxpooling/mean_pred] loss:[mse,w_mse]
ry_branch_setting = {"with_bn":True,"type":"maxpooling","loss":"cls","with_reg":True,"reg_loss":"w_mse"} #type:maxpooling loss:cls(softmax_cross_entropy)

stop_prob_gradient = True
eb_start_step = 1000

w_seg_base = 30.0
w_decay_steps = 1000
w_decay_rate = 0.6
w_seg_min = 0.0

#stop_prob_gradient = False
#eb_start_step = 70000
#
#w_seg_base = 40.0
#w_decay_steps = 4000
#w_decay_rate = 0.9
#w_seg_min = 5.0



extra_branch_xconv_params = [(8, 1, -1, 16 * x),
                             (12, 2, 384, 32 * x),
                             (16, 4, 192, 64 * x),
                             (20, 6, 64, 128 * x)]

extra_branch_fc_params = [(128 * x, 0.0), (64 * x, 0.5)]

# K, D, P, C
xconv_params = [(8, 1, -1, 32 * x),
                (12, 2, 768, 64 * x),
                (16, 2, 384, 128 * x),
                (16, 6, 128, 156 * x)]

# K, D, pts_layer_idx, qrs_layer_idx
xdconv_params = [(16, 6, 3, 2),
                 (12, 4, 2, 1),
                 (8, 4, 1, 0)]

# C, dropout_rate
fc_params = [(32 * x, 0.0), (32 * x, 0.5)]

#sampleing
with_fps = False
eb_with_fps = False

optimizer = 'adam'
epsilon = 1e-3

#imput data
data_dim = 9
use_extra_features = True
data_format = {"pts_xyz":3,"img_xy":2,"extra_features":4}

with_normal_feature = False

with_X_transformation = True

sorting_method = None
keep_remainder = True
