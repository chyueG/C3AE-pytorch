import scipy.io as scio
from datetime import datetime
import cv2
import math
import numpy as np
import os
from multiprocessing import Pool
import random

def cal_age(take,dob):
    birth = datetime.fromordinal(max(int(dob)-366,1))
    if birth.month<7:
        return take-birth.year
    else:
        return take - birth.year-1

def read_mat(mat_path,dataset):
    mat = scio.loadmat(mat_path)
    full_path = mat[dataset][0,0]["full_path"][0]
    dob       = mat[dataset][0,0]["dob"][0]
    gender    = mat[dataset][0,0]["gender"][0]
    photo_taken = mat[dataset][0,0]["photo_taken"][0]
    face_score = mat[dataset][0,0]["face_score"][0]
    scd_face_score = mat[dataset][0,0]["second_face_score"][0]
    age       = list(map(lambda x,y:cal_age(x,y),photo_taken,dob))

    info = {}
    for idx in range(full_path.shape[0]):
        im_name = full_path[idx][0]
        _age    = age[idx]
        _gender = gender[idx]
        info[im_name] = [_age,_gender]
    return info

def read_landmark(mat_path,dataset,landmark_txt,save_txt):
    total_info = read_mat(mat_path,dataset)
    has_face = []
    with open(landmark_txt,"r") as f:
        for line in f.readlines():
            info = line.strip().split(" ")
            im_name = info[0]
            im_name_v2= im_name[10::]
            landmarks = " ".join(info[i] for i in range(1,len(info)))
            age,gender = total_info[im_name_v2]
            #has_face.append(im_name + " "+ str(age) + " "+str(gender)+" "+landmarks)
            has_face.append(im_name_v2 + " " + str(age) + " " + str(gender))
    with open(save_txt,"w+") as f:
        for item in has_face:
            f.write(item+"\n")

def read_txt(txt_path):
    with open(txt_path,"r") as f:
        info = f.readlines()
        return info

def split_train_val(txt):
    ratio = 0.2
    val_info = []
    train_info =[]
    with open(txt,"r") as f:
        all_info = f.readlines()
        val_bool = list(map(lambda x:x%5==0,range(len(all_info))))
        val_idx = list(map(lambda x,y:int(x)*y,val_bool,range(len(val_bool))))
        val_idx = set(val_idx)


        val_info = [all_info[e] for e in val_idx]
        for i in range(len(val_bool)):
            if i not in val_idx:
                train_info.append(all_info[i])


    dir_path = os.path.dirname(txt)
    with open(os.path.join(dir_path,"train.txt"),"w") as f:
        for item in train_info:
            im_info,age,gender = item.strip().split(" ")
            age = int(age)
            if age<100:
                f.write(item)
            else:
                continue
    with open(os.path.join(dir_path,"val.txt"),"w") as f:
        for item in val_info:
            im_info,age,gender = item.strip().split(" ")
            age = int(age)
            if age<100:
                f.write(item)
            else:
                continue


def crop_face(mes):
    im_root = "/imdb-wike/"

    info = mes.strip().split(" ")
    im_name = info[0]
    im = cv2.imread(os.path.join(im_root,im_name))
    h,w,_ = im.shape
    #im_name= im_name[10::]
    dataset,label,name = im_name.split("/")
    x_axis = list(map(lambda x: int(info[x]), [1, 3, 5, 7, 9]))
    xmin = min(x_axis)
    xmax = max(x_axis)
    y_axis = list(map(lambda x: int(info[x]), [2, 4, 6, 8, 10]))
    ymin = min(y_axis)
    ymax = max(y_axis)
    x_interval = (xmax-xmin)//4
    y_interval = (ymax-ymin)//4
    l_x = list(map(lambda x:xmin-x,[x_interval,2*x_interval,3*x_interval]))
    l_y = list(map(lambda x:ymin-x,[y_interval,2*y_interval,3*y_interval]))
    r_x = list(map(lambda x:xmax+x,[x_interval,2*x_interval,3*x_interval]))
    r_y = list(map(lambda x:ymax+x,[y_interval,2*y_interval,3*y_interval]))

    l_x = [e if e>0 else 0 for e in l_x]
    l_y = [e if e>0 else 0 for e in l_y]
    r_x = [e if e<w else w for e in r_x]
    r_y = [e if e<h else h for e in r_y]
    for i in range(len(l_x)):
        cut_im = im[l_y[i]:r_y[i],l_x[i]:r_x[i],:]
        save_path = os.path.join(im_root,"imdb",str(i),label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + "/" + name,cut_im)
        #cv2.rectangle(im,(l_x[i],l_y[i]),(r_x[i],r_y[i]),(0,0,60*i))
    #cv2.imshow("rectange image",im)
    #cv2.waitKey(0)

def affine_face(im_path,x_axis,y_axis):
    im = cv2.imread(im_path)
    mat = np.zeros([500,500,3])
    """
    point_lst = zip(y_axis,x_axis)
    for point in point_lst:
        cv2.circle(im,point,1,(0,0,255),4)
    cv2.imshow("original image",im)
    cv2.waitKey(0)
    """
    le_x = x_axis[0]
    le_y = y_axis[0]
    re_x = x_axis[1]
    re_y = y_axis[1]
    center_x = (le_x + re_x)//2
    center_y = (le_y + re_y)//2

    angle = math.atan2(float(re_y-le_y),float(re_x - le_x))*180.0/math.pi
    rotate_mat = cv2.getRotationMatrix2D((center_x,center_y),angle,scale=1.0)
    dst_im = cv2.warpAffine(im,rotate_mat,(500,500))
    cv2.imshow("affine_im",dst_im)
    cv2.waitKey(0)


def check_file(txt,root):
    label_info = read_txt(txt)
    eff_info = []
    wrong_info = []
    for item in label_info:
        im_info = item.strip().split(" ")[0]
        for i in ["0","1","2"]:
            im_path = os.path.join(root,i,im_info)
            im = cv2.imread(im_path)
            if im is None:
                wrong_info.append(item)
            else:
                eff_info.append(item)
    eff_info = set(eff_info)
    wrong_info = set(wrong_info)
    return eff_info,wrong_info

def write_txt(path,info):
    with open(path,"w+") as f:
        for item in info:
            f.write(item)


def cal_distribution(txt):
    age_range = [10,20,30,40,50,60,70,80,90]
    info = read_txt(txt)
    age_count = {}
    for i in info:
        _,age,_ = i.strip().split(" ")
        age = int(age)
        age_idx = np.floor(age/10.0)
        if age_idx in age_count:
            age_count[age_idx].append(age)
        else:
            age_count[age_idx] = [age]
    return age_count


if __name__ == "__main__":
    txt = ""
    age_count = cal_distribution(txt)

