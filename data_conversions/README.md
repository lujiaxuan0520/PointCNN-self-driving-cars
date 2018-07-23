## Download Dataset

Download KITTI 3D object detection data and organize the folders as follows:

    dataset/KITTI/
    
        data_object_calib/
            training/calib/
            testing/calib/
        
        data_object_velodyne/
            training/velodyne/
            testing/velodyne/
            
        image_2/
            training/image_2/
            testing/image_2/

        label/
            training/label2/
            training/
            val/ss3d_results/
            val/fpnet_val_results/
            testing/rrc_results/
            
         mkdir dataset_generate/
            prepare_trainval_dataset_gt2dbbox.py
            split_data_trainval.py
            image_sets/ss3d_train.txt
            image_sets/ss3d_val.txt
            image_sets/ss3d_val.txt
            image_sets/fpnet_train.txt
            image_sets/fpnet_val.txt
            image_sets/ss3d_trainval.txt
            image_sets/ss3d_test.txt
            

## Usage

### Prepare Frustum pointcloud

config kitti dataset path in prepare_trainval_dataset_gt2dbbox.py
```
python prepare_trainval_dataset_gt2dbbox.py
```

config list file path in split_data_trainval.py
```
python split_data_trainval.py
```

### Prepare PointCNN .h5 file
python prepare_kitti_partseg_data.py -f ../dataset_root/ -d 9
