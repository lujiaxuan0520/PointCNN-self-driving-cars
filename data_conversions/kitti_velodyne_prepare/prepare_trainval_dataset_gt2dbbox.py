import struct
import numpy as np
import plyfile
import math
import os
from PIL import Image

def load_pc_bin(pcbin_files):

    print "\nread pointsfile:",pcbin_files

    point_num = 0
    point_list = []
    normal_list = []

    np_pts = np.fromfile(pcbin_files, dtype=np.float32)
    
    for i in np_pts.reshape((-1, 4)):

        point_list.append(i)

    return point_list

def save_ply(points, colors, filename):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_color = np.array([tuple(c) for c in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, dtype=vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write(filename)

def save_pts(pts,out_file):

    with open(out_file,"w") as f:

        for p in pts:

            f.writelines(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + " " + str(p[6]) + " " + str(p[7]) + " " + str(p[8]) + "\n")

        
def save_seg(seg,out_file):

    with open(out_file,"w") as f:

        for s in seg:

            f.writelines(str(s) + "\n")

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

def load_png(pngfile):

    return np.array(Image.open(pngfile))

def img2camera(pt_c,z,fx,fy,px,py):

    x_c_p = pt_c[0] - xcam
    y_c_p = pt_c[1] - ycam
    z_c = z
    
    x_c = x_c_p*z_c/fx
    y_c = y_c_p*z_c/fy
    
    return [x_c,y_c,z_c]

def camera2img(pt,fx,fy,px,py):

    x = pt[0]
    y = pt[1]
    z = pt[2]

    img_x = fx*x/z + px
    img_y = fy*y/z + py

    return [img_x,img_y]

def clockwiserot(pt,pt_c,deg):

    rx = pt[0] - pt_c[0]
    ry = pt[1] - pt_c[1]
    deg = -deg

    rx_n = math.cos(deg)*rx-math.sin(deg)*ry
    ry_n = math.cos(deg)*ry+math.sin(deg)*rx

    return [rx_n + pt_c[0],ry_n+pt_c[1]]

def ptinbbox(pt,bbox):

    x = pt[0]
    y = pt[1]
    z = pt[2]

    bbox_area = bbox[-1]
    bbox_pts = [bbox[7],bbox[8],bbox[9],bbox[10],bbox[7]]

    if y <= bbox[5] and y >= bbox[6]:

        area = 0

        for i in range(4):

            x1 = bbox_pts[i][0]
            y1 = bbox_pts[i][1]
            x2 = bbox_pts[i+1][0]
            y2 = bbox_pts[i+1][1]

            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2

            area = area + abs(A*x + B*z + C)/2

        #print area,"/",bbox_area
        if abs(bbox_area - area) < 0.1:
            return True
        else:
            return False

    else:
        return False

#Config KITTI path and out path
velo_root = "../data_object_velodyne/training/velodyne/"
image_root = "../image_2/training/image_2/"
label_root = "../label/training/label_2/"
calib_root = "../data_object_calib/training/calib/"

out_root = "./trainval_frustum_xyrgbi_onlycar_onebox_v1/"

offset_y = 0.2
is_saveply = True
is_onebox = True
z_range = [0.0,75.0]

#objlist = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc' ,'DontCare']
#objlist = ['Car','Van','Truck','Pedestrian','Cyclist','Tram']
objlist = ['Car']

out_pts_root = out_root + "pts/"
out_seg_root = out_root + "seg/"
out_ply_root = out_root + "plyshow/"
out_frustum_label_file = out_root + "frustum_label.txt"
out_log_file = out_root + "log.txt"

colorlist = [(255,255,255),(0,255,0),(192,153,110),(188,199,253),(255,45,153),(123,72,131),(0,255,255),(159,0,142),(0,144,161),(118,79,2),(150,0,61),(188,136,0),(132,169,1),(255,167,254),(254,230,0)]

#check the path
if not os.path.exists(out_pts_root):
    print out_pts_root,"Not Exists! Create",out_pts_root
    os.makedirs(out_pts_root)
if not os.path.exists(out_seg_root):
    print out_seg_root,"Not Exists! Create",out_seg_root
    os.makedirs(out_seg_root)
if is_saveply:
    if not os.path.exists(out_ply_root):
        print out_ply_root,"Not Exists! Create",out_ply_root
        os.makedirs(out_ply_root)

#create and clear calib file
frustum_label_file = open(out_frustum_label_file,'w')
frustum_label_file.close()

#create and clear log file
log_file = open(out_log_file,'w')
log_file.close()

velo_list = dir(velo_root)
velo_list.sort()

for velo_file in velo_list:

    name_id = velo_file.strip().split("/")[-1].split(".")[0]

    if int(name_id) < 0:

        continue

    label_file = label_root + name_id + ".txt"
    image_file = image_root + name_id + ".png"
    calib_file = calib_root + name_id + ".txt"

    #read velo bin file
    pc_list = load_pc_bin(velo_file)

    #read rgb image_2 file
    image_rgb = load_png(image_file)

    Camera_P = []

    #read calib file
    with open(calib_file,"r") as f_c:

        for line in f_c.readlines():

            if line.split(':')[0] == "P2":

                Camera_P = line.strip().split(' ')

    xcam = float(Camera_P[3])
    ycam = float(Camera_P[7])
    
    fx = float(Camera_P[1])
    fy = float(Camera_P[6])
    
    #read label file
    ##########################################################################################################
    # type truncated occluded alpha x1     y1     x2     y2     h    w    l    x      y    z     ry          #
    # Car  0.00      0        1.85  387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57        #
    # ry  (x _> z _> -x : 0 -> -pi),(x _> -z _> -x : 0 -> pi)                                                #
    ##########################################################################################################

    label_list = []
    with open(label_file,"r") as f_l:
    
        for line in f_l.readlines():
    
            line_s = line.strip().split(" ")
            
            if line_s[0] in objlist:
                label_list.append(line_s)
    
    #compute 3dbbox
    bbox_list = []
    
    for l in label_list:
    
        print l
    
        bbox_x1 = float(l[4])
        bbox_y1 = float(l[5])
        bbox_x2 = float(l[6])
        bbox_y2 = float(l[7])
    
        l_cx = float(l[11])
        l_cy = float(l[12])
        l_cz = float(l[13])
    
        l_h = float(l[8])
        l_w = float(l[9])
        l_l = float(l[10])
    
        ry = float(l[14])
    
        maxy = l_cy
        miny = l_cy - l_h
    
        pt_c = [l_cx,l_cz]
    
        bbp1 = clockwiserot([l_cx-l_l/2,l_cz+l_w/2],pt_c,ry)
        bbp2 = clockwiserot([l_cx+l_l/2,l_cz+l_w/2],pt_c,ry)
        bbp3 = clockwiserot([l_cx+l_l/2,l_cz-l_w/2],pt_c,ry)
        bbp4 = clockwiserot([l_cx-l_l/2,l_cz-l_w/2],pt_c,ry)
    
        bbox_list.append([l[0],bbox_x1,bbox_y1,bbox_x2,bbox_y2,maxy,miny-offset_y,bbp1,bbp2,bbp3,bbp4,l,[l_cx,l_cy,l_cz],l_w*l_l])

    #save fru
    for k,bbox in enumerate(bbox_list):

        partid = "%03d"%k

        pts_out_file = out_pts_root + name_id + partid + ".pts"
        seg_out_file = out_seg_root + name_id + partid + ".seg"

        print "process part:",partid

        p1_c = img2camera([bbox[1],bbox[2]],1,fx,fy,xcam,ycam)
        p2_c = img2camera([bbox[3],bbox[4]],1,fx,fy,xcam,ycam)
        cp_c = img2camera([(bbox[1]+bbox[3])/2,(bbox[2]+bbox[4])/2],1,fx,fy,xcam,ycam)
    
        pts = []
        seg = []

        if is_saveply and k == 0:
            pts_ply_w = []
            seg_ply_w = []

        if is_saveply: 
            pts_ply = []
            seg_ply = []
    
        for pt in pc_list:
    
            x = -pt[1]
            y = -pt[2]
            z = pt[0]
        
            pt_t = [x,y,z]

            #fru boundary
            b_x1 = z*p1_c[0]
            b_y1 = z*p1_c[1] - offset_y
            b_x2 = z*p2_c[0]
            b_y2 = z*p2_c[1]

            if x >= b_x1 and x <= b_x2 and y >= b_y1 and y <= b_y2 and z >= z_range[0] and z <= z_range[1]:

                if z > 0:

                    xy_img = camera2img(pt_t,fx,fy,xcam,ycam)
    
                    if xy_img[0] >= 0 and xy_img[1] >=0 and xy_img[0] < 1224 and xy_img[1] < 370:
    
                        i_x = int(xy_img[0])
                        i_y = int(xy_img[1])
                        rgb = image_rgb[i_y,i_x]
    
                    else:
                        i_x = min(max(int(xy_img[0]),0),1223)
                        i_y = min(max(int(xy_img[1]),0),369)
                        rgb = [0,0,0]
    
                pts.append([x,y,z,i_x,i_y,rgb[0],rgb[1],rgb[2],pt[3]])
        
                obj_label = 0
                #color = (pt[3]*255,pt[3]*255,pt[3]*255)
                color = (rgb[0],rgb[1],rgb[2])

                if is_saveply:
                    seg_ply_w.append(color)

                if is_onebox:

                    if ptinbbox(pt_t,bbox):

                        obj_label = objlist.index(bbox[0]) + 1
                        color = colorlist[obj_label]
                else:

                    for bbox_iter in bbox_list:

                        if ptinbbox(pt_t,bbox_iter):
                            obj_label = objlist.index(bbox_iter[0]) + 1
                            color = colorlist[obj_label]
                
                seg.append(obj_label)
                
                if is_saveply:
                    seg_ply.append(color)
                    

        label_map,label_count = np.unique(seg,return_counts=True)

        print "label_map:",label_map

        log_file = open(out_log_file,'a')
        log_file.writelines("labelmap_" + name_id + partid + "," + str(label_map) + "," +str(label_count) + "," + str(len(seg)) + "\n")
        log_file.close()
        
        #To Frustum coordinate
        z_direction = cp_c
        c_pt = bbox[-2]
        frustum_rot = math.atan(-z_direction[0]/z_direction[2])
        c_pt_f = clockwiserot([c_pt[0],c_pt[2]],[0,0],frustum_rot)
        c_pc_f = clockwiserot([cp_c[0],cp_c[2]],[0,0],frustum_rot)

        for pt in pts:

            pt_f = clockwiserot([pt[0],pt[2]],[0,0],frustum_rot)

            if is_saveply: 
                
                pts_ply.append([pt_f[0],pt[1],pt_f[1]])
                pts_ply_w.append([pt[0],pt[1],pt[2]])

            pt[0] = pt_f[0]
            pt[2] = pt_f[1]

            
        #record coordinates label
        ori_label = bbox[-3]
        frustum_label_file = open(out_frustum_label_file,'a')
        #type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
        frustum_label_file.writelines("ori_"   + name_id + partid)
        for l_ele in ori_label:
            frustum_label_file.writelines(" "   + l_ele)
        frustum_label_file.writelines("\n")

        frustum_label_file.writelines("fru_"   + name_id + partid)
        for l_id,l_ele in enumerate(ori_label):
            if l_id == 11:
                frustum_label_file.writelines(" "   + str(round(c_pt_f[0],6)))
            elif l_id == 13:
                frustum_label_file.writelines(" "   + str(round(c_pt_f[1],6)))
            elif l_id == 14:
                temp = round(float(l_ele)+frustum_rot,6)

                if temp > math.pi:
                    ry = (temp - 2 * math.pi)

                elif temp < -math.pi:
                    ry = (temp + 2 * math.pi)

                else:
                    ry = temp

                frustum_label_file.writelines(" " + str(ry))
            else:
                frustum_label_file.writelines(" "   + l_ele)
        frustum_label_file.writelines("\n")

        frustum_label_file.writelines("calib_" + name_id + partid + " " + str(frustum_rot) + "\n")
        frustum_label_file.writelines("caxis_" + name_id + partid + " " + str(c_pc_f[0]) + " " + str(cp_c[1]) + " " + str(c_pc_f[1]) + "\n")
        frustum_label_file.close()
        print z_direction,frustum_rot,"append to",out_frustum_label_file

        if is_saveply: 
            ply_out_file = out_ply_root + name_id + partid + ".ply"
            print "save ply:",ply_out_file
            save_ply(pts_ply,seg_ply,ply_out_file)

        #save pts
        print "save pts:",pts_out_file
        save_pts(pts,pts_out_file)
        
        #save seg
        print "save seg:",seg_out_file
        save_seg(seg,seg_out_file)
    
    if is_saveply and len(label_list) != 0:
        ply_out_file = out_ply_root + name_id + ".ply"
        print "save ply:",ply_out_file
        save_ply(pts_ply_w,seg_ply_w,ply_out_file)
    