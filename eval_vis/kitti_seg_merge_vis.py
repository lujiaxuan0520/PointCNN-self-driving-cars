import struct
import numpy as np
import plyfile
import math
import os
from PIL import Image

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

def color2rgb(color):

    r = ((float(color[0]) + 1.0) * 255.0)/2
    g = ((float(color[1]) + 1.0) * 255.0)/2
    b = ((float(color[2]) + 1.0) * 255.0)/2

    return (r,g,b)

def load_pc_bin(pcbin_files):

    #print "read pointsfile:",pcbin_files

    point_num = 0
    point_list = []
    normal_list = []

    np_pts = np.fromfile(pcbin_files, dtype=np.float32)
    
    for i in np_pts.reshape((-1, 4)):

        point_list.append(i)

    return point_list

def load_png(pngfile):

    return np.array(Image.open(pngfile))

def save_pts(pts,out_file):

    with open(out_file,"w") as f:

        for p in pts:

            f.writelines(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + " " + str(p[6]) + " " + str(p[7]) + " " + str(p[8]) + "\n")
        
def save_seg(seg,out_file):

    with open(out_file,"w") as f:

        for s in seg:

            f.writelines(str(s) + "\n")

def clockwiserot(pt,pt_c,deg):

    rx = pt[0] - pt_c[0]
    ry = pt[1] - pt_c[1]
    deg = -deg

    rx_n = math.cos(deg)*rx-math.sin(deg)*ry
    ry_n = math.cos(deg)*ry+math.sin(deg)*rx

    return [rx_n + pt_c[0],ry_n+pt_c[1]]

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

def pointline(pt1,pt2,interval = 0.1):

    line = []

    x1 = pt1[0]
    y1 = pt1[1]
    z1 = pt1[2]

    x2 = pt2[0]
    y2 = pt2[1]
    z2 = pt2[2]

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    num_x = abs(dx/interval)
    num_y = abs(dy/interval)
    num_z = abs(dz/interval)

    num = int(max(num_x,num_y,num_z))

    for i in range(num):

        line.append([x1 + i*dx/num,y1 + i*dy/num,z1 + i*dz/num])

    line.append([x2,y2,z2])

    return line

def pointline_list(pt1,pt_list,interval = 0.1):

    line = []

    for pt2 in pt_list:

        line_t = pointline(pt1,pt2,interval)

        line = line + line_t

    return line

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

############################################ arguments ############################################
#pred root
seg_label_root = "./val_label/"
seg_data_root = "./val_data/"
seg_pred_root = "./pred_4/seg/"
frustum_label_file_path = "./frustum_label.txt"

#kitti root
velo_root = "../../KITTI/data_object_velodyne/training/velodyne/"
label_root = "../../KITTI/label_2/"
calib_root = "../../KITTI/data_object_calib/training/calib/"
image_root = "../../KITTI/image_2/training/image_2/"

#out dir
out_root = "./seg_merge_ply_vis/"
out_log_file = out_root + "log_seg_merge.txt"

is_save_part_ply = True

is_vis_frub = True
frub_color = (100,255,255)

is_vis_bbox = True
bbox_color = (255,100,230)

offset_y = 0.2

############################################ load source ############################################
#check the output path
if not os.path.exists(out_root):
    print out_root,"Not Exists! Create",out_root
    os.makedirs(out_root)

#create and clear log file
log_file = open(out_log_file,'w')
log_file.close()
log_file = open(out_log_file,'a')

#get pred_list
pred_list = dir(seg_pred_root)
pred_list.sort()

#get scene_list
scene_list = np.unique([pred_file.strip().split("/")[-1].split(".")[0][0:6] for pred_file in pred_list])
scene_list.sort()

#load frustum label
frustum_label_dic = {}
with open(frustum_label_file_path,"r") as f:

    for line in f.readlines():

        line_s = line.strip().split(" ")

        if line_s[0].split("_")[0] == "calib":

            frustum_label_dic[line_s[0].split("_")[1]] = float(line_s[1].strip())

pts_num_all = 0
acc_num_all = 0

for scene_id in scene_list:

    print "process:",scene_id
    log_file.writelines("process scene:" + scene_id + "\n")

    pts_num_scene = 0
    acc_num_scene = 0

    fru_pred_list = [pred_file for pred_file in pred_list if pred_file.strip().split("/")[-1].split(".")[0][0:6] == scene_id]

    pts_scene = []
    seg_scene = []
    sgt_scene = []
    color_pr_scene = []
    color_gt_scene = []
    color_er_scene = []

    #scene ori ply
    velo_file  = velo_root  + scene_id + ".bin"
    image_file = image_root + scene_id + ".png"
    calib_file = calib_root + scene_id + ".txt"
    label_file = label_root + scene_id + ".txt"

    #read velo bin file
    pc_list = load_pc_bin(velo_file)

    #read rgb image file
    image_rgb = load_png(image_file)

    #read calib file
    Camera_P = []
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
    #objlist = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc' ,'DontCare']
    #objlist = ['Car','Van','Truck','Pedestrian','Cyclist','Tram']

    objlist = ['Car']
    colorlist = [(255,255,255),(0,255,0),(192,153,110),(188,199,253),(255,45,153),(123,72,131),(0,255,255),(159,0,142),(0,144,161),(118,79,2),(150,0,61),(188,136,0),(132,169,1),(255,167,254),(254,230,0)]
    label_list = []
    
    with open(label_file,"r") as f_l:
    
        for line in f_l.readlines():
            line_s = line.strip().split(" ")
            
            if line_s[0] in objlist:
                label_list.append(line_s)

    ############################################ save ori ply ############################################
    pts_ply_ori = []
    seg_ply_ori = []

    for pt in pc_list:
    
        x = -pt[1]
        y = -pt[2]
        z = pt[0]
        
        pt_t = [x,y,z]

        #get color from img
        if z > 0:

            xy_img = camera2img(pt_t,fx,fy,xcam,ycam)
    
            if xy_img[0] >= 0 and xy_img[1] >= 0 and xy_img[0] < 1224 and xy_img[1] < 370:
    
                i_x = int(xy_img[0])
                i_y = int(xy_img[1])
                rgb = image_rgb[i_y,i_x]
    
            else:
                rgb = [0,0,0]
        
            #color = (pt[3]*255,pt[3]*255,pt[3]*255)
            color = (rgb[0],rgb[1],rgb[2])

            #pts_ply_ori.append([x,y,z,i_x,i_y,rgb[0],rgb[1],rgb[2],pt[3]])
            pts_ply_ori.append(pt_t)
            seg_ply_ori.append(color)

    #save ori ply
    save_ply(pts_ply_ori,seg_ply_ori,out_root + scene_id + "_ori.ply")

    ############################################# save gt ply ############################################
    #compute bbox
    bbox_list = []
    
    for l in label_list:
    
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

    #label gt
    if is_vis_frub:
        pts_ply_frub = []
        seg_ply_frub = []
    if is_vis_bbox:
        pts_ply_bbox = []
        seg_ply_bbox = []

    pts_ply_gt = []
    seg_ply_gt = []

    for k,bbox in enumerate(bbox_list):

        p1_c = img2camera([bbox[1],bbox[2]],1,fx,fy,xcam,ycam)
        p2_c = img2camera([bbox[3],bbox[4]],1,fx,fy,xcam,ycam)
        cp_c = img2camera([(bbox[1]+bbox[3])/2,(bbox[2]+bbox[4])/2],1,fx,fy,xcam,ycam)

        if is_vis_frub:
            z_list = []

        #vis bbox
        if is_vis_bbox:

            bbox_maxy = bbox[5]
            bbox_miny = bbox[6]

            bbox_xz = [bbox[7],bbox[8],bbox[9],bbox[10]]

            bbox_center = bbox[-2]

            #vis center
            pts_ply_bbox.append(bbox_center)
            seg_ply_bbox.append(bbox_color)

            #vis bbox
            p1 = [bbox_xz[0][0],bbox_miny,bbox_xz[0][1]]
            p2 = [bbox_xz[1][0],bbox_miny,bbox_xz[1][1]]
            p3 = [bbox_xz[2][0],bbox_miny,bbox_xz[2][1]]
            p4 = [bbox_xz[3][0],bbox_miny,bbox_xz[3][1]]
            p5 = [bbox_xz[0][0],bbox_maxy,bbox_xz[0][1]]
            p6 = [bbox_xz[1][0],bbox_maxy,bbox_xz[1][1]]
            p7 = [bbox_xz[2][0],bbox_maxy,bbox_xz[2][1]]
            p8 = [bbox_xz[3][0],bbox_maxy,bbox_xz[3][1]]

            line = []
            line_t = pointline_list(p1,[p5,p2,p4])
            line = line + line_t
            line_t = pointline_list(p6,[p2,p5,p7])
            line = line + line_t
            line_t = pointline_list(p3,[p2,p4,p7])
            line = line + line_t
            line_t = pointline_list(p8,[p5,p4,p7])
            line = line + line_t

            for lp in line:

                pts_ply_bbox.append(lp)
                seg_ply_bbox.append(bbox_color)

        for i,pt in enumerate(pts_ply_ori):

            color = seg_ply_ori[i]

            if z >= 0:

                z = pt[2]

                b_x1 = z*p1_c[0]
                b_y1 = z*p1_c[1] - offset_y
                b_x2 = z*p2_c[0]
                b_y2 = z*p2_c[1]

                #vis fru
                if is_vis_frub and round(z,1) not in z_list:

                    pts_ply_frub.append([b_x1,b_y1,z])
                    seg_ply_frub.append(frub_color)

                    pts_ply_frub.append([b_x1,b_y2,z])
                    seg_ply_frub.append(frub_color)

                    pts_ply_frub.append([b_x2,b_y1,z])
                    seg_ply_frub.append(frub_color)

                    pts_ply_frub.append([b_x2,b_y2,z])
                    seg_ply_frub.append(frub_color)

                    z_list.append(round(z,1))


                obj_label = 0

                if ptinbbox(pt,bbox):

                    obj_label = objlist.index(bbox[0]) + 1
                    color = colorlist[obj_label]

                    pts_ply_gt.append(pt)
                    seg_ply_gt.append(color)

    #save ori ply
    pts_ply_gt_save = pts_ply_gt + pts_ply_ori
    seg_ply_gt_save = seg_ply_gt + seg_ply_ori

    if is_vis_frub:
        pts_ply_gt_save = pts_ply_frub + pts_ply_gt_save
        seg_ply_gt_save = seg_ply_frub + seg_ply_gt_save
    if is_vis_bbox:
        pts_ply_gt_save = pts_ply_bbox + pts_ply_gt_save
        seg_ply_gt_save = seg_ply_bbox + seg_ply_gt_save

    save_ply(pts_ply_gt_save,seg_ply_gt_save,out_root + scene_id + "_gt.ply")                

    ############################################ save pred ply ############################################

    pts_ply_pred = []
    seg_ply_pred = []

    for fru_pred_file in fru_pred_list:

        pts_num = 0
        acc_num = 0

        file_id = fru_pred_file.strip().split("/")[-1].split(".")[0]
        part_id = file_id[6:9]

        log_file.writelines("\tfru_acc:" + file_id)

        seg_data_file = seg_data_root + file_id + ".pts"
        seg_label_file = seg_label_root + file_id + ".seg"

        #read pts
        fru_pts = [pt.strip().split(" ")[0:3] for pt in open(seg_data_file, 'r').readlines()]
        fru_rgbi = [pt.strip().split(" ")[3:7] for pt in open(seg_data_file, 'r').readlines()]

        #read seg label
        fru_seg_gt = [int(seg.strip()) for seg in open(seg_label_file, 'r').readlines()]

        #read seg pred
        fru_seg_pred = [int(seg.strip()) for seg in open(fru_pred_file, 'r').readlines()]

        #save part ply
        if is_save_part_ply:

            pts_part = []
            seg_part = []
            sgt_part = []
            color_pr_part = []
            color_gt_part = []
            color_er_part = []

        for i,pt in enumerate(fru_pts):

            pt_f = [float(pt[0]),float(pt[1]),float(pt[2])]

            if is_save_part_ply:
            
                pts_part.append(pt_f)
                seg_part.append(fru_seg_pred[i])
                sgt_part.append(fru_seg_gt[i])
            
                if fru_seg_pred[i] == 1:
                    color_pr_part.append((0,255,0))
                else:
                    color_pr_part.append(color2rgb(fru_rgbi[i]))

                if fru_seg_gt[i] == 1:
                    color_gt_part.append((0,255,0))
                else:
                    color_gt_part.append(color2rgb(fru_rgbi[i]))

                if fru_seg_pred[i] != fru_seg_gt[i]:
                    if fru_seg_gt[i] == 0:
                        color_er_part.append((225,255,0))
                    elif fru_seg_gt[i] == 1:
                        color_er_part.append((255,0,0))
                else:
                    if fru_seg_gt[i] == 1:
                        color_er_part.append((0,255,0))
                    else:
                        color_er_part.append(color2rgb(fru_rgbi[i]))

            #TODO REROT
            if fru_seg_pred[i] != fru_seg_gt[i]:

                pt_o_xz = clockwiserot([pt_f[0],pt_f[2]],[0,0],-frustum_label_dic[file_id])
                pt_o = [pt_o_xz[0],pt_f[1],pt_o_xz[1]]

                if fru_seg_gt[i] == 0:
                    pts_ply_pred.append(pt_o)
                    seg_ply_pred.append((225,255,0))

                elif fru_seg_gt[i] == 1:
                    pts_ply_pred.append(pt_o)
                    seg_ply_pred.append((255,0,0))

            elif fru_seg_pred[i] == 1:

                pt_o_xz = clockwiserot([pt_f[0],pt_f[2]],[0,0],-frustum_label_dic[file_id])
                pt_o = [pt_o_xz[0],pt_f[1],pt_o_xz[1]]

                pts_ply_pred.append(pt_o)
                seg_ply_pred.append((0,255,0))

        #save pred ply
        pts_ply_gt_save = pts_ply_pred + pts_ply_ori
        seg_ply_gt_save = seg_ply_pred + seg_ply_ori

        if is_vis_frub:
            pts_ply_gt_save = pts_ply_frub + pts_ply_gt_save
            seg_ply_gt_save = seg_ply_frub + seg_ply_gt_save
        if is_vis_bbox:
            pts_ply_gt_save = pts_ply_bbox + pts_ply_gt_save
            seg_ply_gt_save = seg_ply_bbox + seg_ply_gt_save

        save_ply(pts_ply_gt_save,seg_ply_gt_save,out_root + scene_id + "_pred.ply")

        if is_save_part_ply:
            #save part pred ply
            save_ply(pts_part,color_pr_part,out_root + file_id + "_p.ply")
            #save part gt ply
            save_ply(pts_part,color_gt_part,out_root + file_id + "_g.ply")
            #save part err ply
            save_ply(pts_part,color_er_part,out_root + file_id + "_e.ply")
            
        #compute acc
        pts_num = len(fru_seg_gt)

        for k,seg_gt in enumerate(fru_seg_gt):

            if seg_gt == fru_seg_pred[k]:

                acc_num = acc_num + 1

        pts_num_scene = pts_num_scene + pts_num
        acc_num_scene = acc_num_scene + acc_num

        acc_p = acc_num*1.0/pts_num*1.0
        if acc_p < 0.9:
            log_file.writelines(" " + str(acc_num) + "/" + str(pts_num) + " " + str(acc_p) + " " + "low_acc" + "\n")
        else:
            log_file.writelines(" " + str(acc_num) + "/" + str(pts_num) + " " + str(acc_p) + "\n")


    log_file.writelines("\tscene_acc:" + str(acc_num_scene) + "/" + str(pts_num_scene) + " " + str(acc_num_scene*1.0/pts_num_scene*1.0) + "\n")

    pts_num_all = pts_num_all + pts_num_scene
    acc_num_all = acc_num_all + acc_num_scene

log_file.writelines("acc:" + str(acc_num_all) + "/" + str(pts_num_all) + " " + str(acc_num_all*1.0/pts_num_all*1.0) + "\n")

