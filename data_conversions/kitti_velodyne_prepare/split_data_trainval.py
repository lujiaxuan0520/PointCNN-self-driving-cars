import os
import shutil

#list file
train_list_file = "./image_sets/ss3d_train.txt"
val_list_file = "./image_sets/ss3d_val.txt"

#train_list_file = "./image_sets/fpnet_train.txt"
#val_list_file = "./image_sets/fpnet_val.txt"

#source root
source_root = './trainval_frustum_xyrgbi_onlycar_onebox_v1/'
img_source_root = '../image_2/training/image_2/'

#target root
target_root = './kitti_dataset_xyrgbi_onlycar_onebox_trainval_ss3d_v1/'

seg_source_root = source_root + "seg/"
pts_source_root = source_root + "pts/"

pts_train_root = target_root + 'train_data/'
seg_train_root = target_root + 'train_label/'
img_train_root = target_root + 'train_img/'

pts_val_root = target_root + 'val_data/'
seg_val_root = target_root + 'val_label/'
img_val_root = target_root + 'val_img/'

min_p = 6

def dir(root,type = 'f',addroot = True):

    dirList = []
    fileList = []

    files = os.listdir(root)  

    for f in files:
        if(os.path.isdir(root + f)):  
            if addroot == True:
                dirList.append(root + f)
            else:
                dirList.append(f)

        if(os.path.isfile(root + f)):          
            if addroot == True:           
                fileList.append(root + f)
            else:
                fileList.append(f)

    if type == "f":
        return fileList

    elif type == "d":
        return dirList

    else:
        print("ERROR: TMC.dir(root,type) type must be [f] for file or [d] for dir")

        return 0

def check_copy_pts(source,target,minpt,c):

    with open(source,"r") as f:

        lines = f.readlines()

        if len(lines) < minpt:

            return False

        else:

            pts = []

            for line in lines:

                line_s = line.strip().split(" ")[0:c]

                pt = ""

                for d in line_s:

                    pt = pt + d + " "

                pts.append(pt.strip())

    with open(target,"w") as f:

        for pt in pts:

            f.writelines(pt + "\n")

    return True

def check_copy_pts_rgb(source,target,minpt,c):

    with open(source,"r") as f:

        lines = f.readlines()

        if len(lines) < minpt:

            return False

        else:

            pts = []

            for line in lines:

                line_s = line.strip().split(" ")[0:c]

                pt = ""

                for k,d in enumerate(line_s):

                    if k in [5,6,7]:

                        d_c = str(int(d)*2.0/255.0 - 1.0)

                        pt = pt + d_c + " "

                    elif k == 8:

                        d_i = str(float(d)*2.0 - 1.0)

                        pt = pt + d_i + " "

                    else:

                        pt = pt + d + " "

                pts.append(pt.strip())

    with open(target,"w") as f:

        for pt in pts:

            f.writelines(pt + "\n")

    return True

if not os.path.exists(pts_train_root):
    print(pts_train_root,"Not Exists! Create",pts_train_root)
    os.makedirs(pts_train_root)
if not os.path.exists(seg_train_root):
    print(seg_train_root,"Not Exists! Create",seg_train_root)
    os.makedirs(seg_train_root)
if not os.path.exists(img_train_root):
    print(img_train_root,"Not Exists! Create",img_train_root)
    os.makedirs(img_train_root)

if not os.path.exists(pts_val_root):
    print(pts_val_root,"Not Exists! Create",pts_val_root)
    os.makedirs(pts_val_root)
if not os.path.exists(seg_val_root):
    print(seg_val_root,"Not Exists! Create",seg_val_root)
    os.makedirs(seg_val_root)
if not os.path.exists(img_val_root):
    print(img_val_root,"Not Exists! Create",img_val_root)
    os.makedirs(img_val_root)

train_list = []
val_list = []

with open(train_list_file,"r") as f_l:
    for line in f_l.readlines():
        train_list.append(line.strip())

with open(val_list_file,"r") as f_l:
    for line in f_l.readlines():
        val_list.append(line.strip())

pts_list = dir(pts_source_root)
part_name = []

for pts_path in pts_list:
    part_name.append(pts_path.strip().split("/")[-1].split(".")[0])

#cp calib
shutil.copy(source_root + 'frustum_label.txt', target_root + 'frustum_label.txt')

#cp train
for name in train_list:

    print("process:",name)

    #cp img
    source_img_name = img_source_root + name + '.png'
    target_img_name = img_train_root + name + '.png'
    shutil.copy(source_img_name, target_img_name)
    print('cp ' + source_img_name + ' ' + source_img_name)

    for part in part_name:

        if part[0:6] == name:

            source_pts_name = pts_source_root + part + ".pts"
            source_seg_name = seg_source_root + part + ".seg"

            target_pts_name = pts_train_root + part + ".pts"
            target_seg_name = seg_train_root + part + ".seg"

            print("part:",part)

            #copy
            if check_copy_pts_rgb(source_pts_name,target_pts_name,min_p,9):
                print('cp ' + source_pts_name + ' ' + target_pts_name)
                shutil.copy(source_seg_name, target_seg_name) 
                print('cp ' + source_seg_name + ' ' + target_seg_name)            

#cp val
for name in val_list:

    print("process:",name)

    #cp img
    source_img_name = img_source_root + name + '.png'
    target_img_name = img_val_root + name + '.png'
    shutil.copy(source_img_name, target_img_name)
    print('cp ' + source_img_name + ' ' + source_img_name)

    for part in part_name:

        if part[0:6] == name:

            source_pts_name = pts_source_root + part + ".pts"
            source_seg_name = seg_source_root + part + ".seg"

            target_pts_name = pts_val_root + part + ".pts"
            target_seg_name = seg_val_root + part + ".seg"

            print("part:",part)

            #copy
            if check_copy_pts_rgb(source_pts_name,target_pts_name,min_p,9):
                print('cp ' + source_pts_name + ' ' + target_pts_name)
                shutil.copy(source_seg_name, target_seg_name)
                print('cp ' + source_seg_name + ' ' + target_seg_name)