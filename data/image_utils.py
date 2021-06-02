from skimage import color
from PIL import Image

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

from skimage import io
from skimage import img_as_ubyte

fine_size = 256
l_cent = 50.
l_norm = 100.
ab_norm = 110.

def img_transform(im):
    # img: PIL object type
    # output: transformed image

    # define transform
    T = transforms.Compose([transforms.Resize((fine_size, fine_size), interpolation=2),
							transforms.ToTensor()])
    output = T(im)
    return output

def out_full_data(img_path):
    rgb_img, gray_img = get_gray_rgb_pil(img_path)
    rgb_img = img_transform(rgb_img)
    gray_img = img_transform(gray_img)
    rgb_lab_img = get_lab_img(rgb_img)
    gray_lab_img = get_lab_img(gray_img)

    # return gray image with size H*W, and lab image with first two channel.
    return gray_lab_img['A'], rgb_lab_img['B']


def get_gray_rgb_pil(img_path):
    # img_path: input image path
    # rgb, gray: rgb image, gray image (PIL.Image type)
    rgb = Image.open(img_path)
    if len(np.asarray(rgb).shape) == 2:
        rgb = np.stack([np.asarray(rgb), np.asarray(rgb), np.asarray(rgb)], 2)
        rgb = Image.fromarray(rgb)
    gray = np.round(color.rgb2gray(np.asarray(rgb)) * 255.0).astype(np.uint8)
    gray = np.stack([gray, gray, gray], -1)
    gray = Image.fromarray(gray)

    return rgb, gray

def get_lab_img(img):
    lab_data = {}
    lab_img = rgb2lab(img)
    lab_data['A'] = lab_img[0,:,:]
    lab_data['B'] = lab_img[1:,:,:]

    #out =  add_rand_color_patches(lab_img, args, p=p, num_points=num_points)
    return lab_data


# Color conversion code
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
    out = torch.cat((x[None,:,:],y[None,:,:],z[None,:,:]),dim=0)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[0,:,:]-1.53715152*xyz[1,:,:]-0.49853633*xyz[2,:,:]
    g = -0.96925495*xyz[0,:,:]+1.87599*xyz[1,:,:]+.04155593*xyz[2,:,:]
    b = .05564664*xyz[0,:,:]-.20404134*xyz[1,:,:]+1.05731107*xyz[2,:,:]

    rgb = torch.cat((r[None,:,:],g[None,:,:],b[None,:,:]),dim=0)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[1,:,:]-16.
    a = 500.*(xyz_int[0,:,:]-xyz_int[1,:,:])
    b = 200.*(xyz_int[1,:,:]-xyz_int[2,:,:])
    out = torch.cat((L[None,:,:],a[None,:,:],b[None,:,:]),dim=0)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[0,:,:]+16.)/116.
    x_int = (lab[1,:,:]/500.) + y_int
    z_int = y_int - (lab[2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[None,:,:],y_int[None,:,:],z_int[None,:,:]),dim=0)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out

def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    # print(lab[0, 0, 0, 0])
    l_rs = (lab[[0],:,:] - l_cent) / l_norm
    # print(l_rs[0, 0, 0, 0])
    ab_rs = lab[1:,:,:] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=0)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs):
    l = lab_rs[[0],:,:] * l_norm + l_cent
    ab = lab_rs[1:,:,:] * ab_norm
    lab = torch.cat((l, ab), dim=0)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

## Save image to file
# Save lab image to file
def save_image_from_tensor(path, lab_img):
    img = torch.clamp(lab2rgb(lab_img), 0.0, 1.0)
    img = np.transpose(img.data.numpy(), (1, 2, 0))

    io.imsave(path, img_as_ubyte(img))