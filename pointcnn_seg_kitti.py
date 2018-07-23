from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pointcnn_kitti import PointCNN


class Net(PointCNN):
    def __init__(self, points, features, images, image_xy, num_class, is_training, setting):
        PointCNN.__init__(self, points, features, images, image_xy, num_class, is_training, setting, 'segmentation')
