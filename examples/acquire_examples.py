import os
import sys
import argparse
import random
import shutil

def get_examples(pre_path,post_path,save_path,num_img):
    pre_list = os.listdir(pre_path)
    post_list = os.listdir(post_path)

    pre_list_len = len(pre_list)
    post_list_len = len(post_list)
    total = min(pre_list_len,post_list_len)
    count = num_img
    prev_ind = []

    save_pre_path = os.path.join(save_path,"original")
    save_post_path = os.path.join(save_path,"standardized")

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as oserr:
            print("Error: "+str(oserr))
    try:
        os.mkdir(save_pre_path)
        os.mkdir(save_post_path)
    except OSError as oserr:
        print("Error: "+str(oserr))
    while count>0:
        random_img = random.randint(0,total-1)
        if random_img in prev_ind:
            # guarantee unique image id's
            continue
        img_root = os.path.splitext(pre_list[random_img])[0]
        pre_img = img_root+".png"
        post_img = img_root+"-0.png"
        if not os.path.exists(os.path.join(save_pre_path,pre_img)):
            dest_pre = shutil.copy2(os.path.join(pre_path,pre_img),os.path.join(save_pre_path,pre_img))
        if not os.path.exists(os.path.join(save_post_path,post_img)):
            dest_post = shutil.copy2(os.path.join(post_path,post_img),os.path.join(save_post_path,post_img))
        prev_ind.append(random_img)
        count -= 1
    return



if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Returns pairs of before processing and post standardization images and stores into a folder')
    parser.add_argument('--pre_standardization_path',dest='pre_path',default='.',type=str,help='Pre-standardization images')
    parser.add_argument('--post_standardization_path',dest='post_path',default='.',type=str,help='Post-standardization images')
    parser.add_argument('--save_path',dest='save_path',default='.',type=str,help='Save path')
    parser.add_argument('--n',dest='num_img',default = 3,type=int,help='Number of images pairs')

    args = parser.parse_args()
    get_examples(args.pre_path,args.post_path,args.save_path,args.num_img)
