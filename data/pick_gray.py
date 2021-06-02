import cv2
import os
import torch
import shutil
from image_utils import *

ab_norm = 110.
l_cent = 50.
l_norm = 100.

def pick_gray(data_raw, ab_thresh=5., p=.125, num_points=None):
    data = {}
    data_lab = rgb2lab(data_raw[0])
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if mask == 0:
            return 1
        #if(torch.sum(mask)==0):
        #    return None


def pick(img_path):
    x = os.listdir(img_path)
    p1, p2 = 1, 1
    thre1, thre2 = 10, 0
    gray_list = []
    count = 1
    new_path = '/home/ubuntu/576_Project_Colorization/gray_imgs/valid/'
    for i in x:
        print(count)
        img_path2 = os.path.join(img_path, i)
        new_path2 = os.path.join(new_path, i)
        rgb_img, gray_img = get_gray_rgb_pil(img_path2)
        rgb_img = [img_transform(rgb_img)]

        t = pick_gray(rgb_img, thre1, p1)
        count += 1
        if t == 1:
            shutil.move(img_path2, new_path2)

def lab2rgb(lab_rs):
    l = lab_rs[:,[0],:,:].l_norm + l_cent
    ab = lab_rs[:,1:,:,:].ab_norm
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    # print(lab[0, 0, 0, 0])
    l_rs = (lab[:,[0],:,:]-l_cent)/l_norm
    # print(l_rs[0, 0, 0, 0])
    ab_rs = lab[:,1:,:,:]/ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])
    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[0,:,:]+.357580*rgb[1,:,:]+.180423*rgb[2,:,:]
    y = .212671*rgb[0,:,:]+.715160*rgb[1,:,:]+.072169*rgb[2,:,:]
    z = .019334*rgb[0,:,:]+.119193*rgb[1,:,:]+.950227*rgb[2,:,:]
    out = torch.cat((x[None,:,:],y[None,:,:],z[None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out


if __name__ == '__main__':
    img_path = '/home/ubuntu/576_Project_Colorization/img_data/coco/valid/'
    pick(img_path)
