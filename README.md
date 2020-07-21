# PointCNN-self-driving-cars
本项目为2018年山东大学第四届“可视计算”暑期学校无人车小组的DIY项目，参考山东大学提出的基于点云的PointCNN点卷积神经网络，使用KITTI数据集对车辆的点云输入进行语义分割，并就划分出的车辆图像进行三维包装盒的预测，以帮助无人车定位车辆的三维位置。

代码结构说明：
+ data_utils.py -> 数据处理相关函数
+ pointcnn_kitti.py -> PointCNN网络结构
+ pointfly.py -> 项目所使用的函数
+ test_seg_kitti.py -> 测试函数
+ train_val_seg_kitti.py -> 训练与验证函数
