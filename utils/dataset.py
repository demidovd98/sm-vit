import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


# My:
from torchvision import transforms
from torchvision.utils import save_image

import random
from torchvision.transforms import functional as F

import U2Net
from U2Net.u2net_test import mask_hw

from skimage import transform as transform_sk
import gc
#



class Generic_smvit_DS():


    def generic_preprocess(self, file_list, file_list_full, shape_hw_list, data_len, train_test_list=None):

        img = []
        mask = []

        ## For other experiments:
        if self.ds_name != "CUB":
            self.gt_bbox = False
            self.gt_parts = False

        if self.gt_bbox:
            bounding_boxes_file = open(os.path.join(self.root, 'bounding_boxes.txt'))

            bb_list = []
            for line in bounding_boxes_file:
                bb_list_x = line[:-1].split(' ')[-4]
                bb_list_y = line[:-1].split(' ')[-3]
                bb_list_w = line[:-1].split(' ')[-2]
                bb_list_h = line[:-1].split(' ')[-1]

                bb_list.append( [ int(bb_list_x.split('.')[0]),
                                    int(bb_list_y.split('.')[0]),
                                    int(bb_list_w.split('.')[0]),
                                    int(bb_list_h.split('.')[0]) ]
                                    )

            bb_list = [x for i, x in zip(train_test_list, bb_list) if i]

        if self.gt_parts:
            parts_file = open(os.path.join(self.root, 'parts/part_locs.txt'))

            PARTS_NUM = 15
            parts_list = []
            part_t = []
            part_count = 0

            for line in parts_file:
                part_t_raw_x = line[:-1].split(' ')[-3]
                part_t_raw_y = line[:-1].split(' ')[-2]
                part_t_pres = line[:-1].split(' ')[-1]

                part_t.append ( [ int(part_t_pres),
                                    int(part_t_raw_x.split('.')[0]),
                                    int(part_t_raw_y.split('.')[0]) ]
                                    )
                part_count = part_count + 1

                if (part_count >= PARTS_NUM):
                    parts_list.append( part_t )
                    part_t = []
                    part_count = 0

            parts_list = [x for i, x in zip(train_test_list, parts_list) if i]
        ##


        print(f'[INFO] Pre-processing {self.mode} files...')

        if self.sm_vit:
            if self.full_ds:
                mask_u2n_list, x_u2n_list, y_u2n_list, h_u2n_list, w_u2n_list = \
                    mask_hw(full_ds=self.full_ds, img_path=file_list_full, shape_hw=shape_hw_list)
            else: # for debug
                img_path = os.path.join(self.root, self.base_folder, file_list)
                img_temp = scipy.misc.imread(img_path)
                h_max_temp = img_temp.shape[0] # y
                w_max_temp = img_temp.shape[1] # x
                mask_u2n, x_u2n, y_u2n, h_u2n, w_u2n = \
                    mask_hw(full_ds=self.full_ds, img_path=img_path, shape_hw=(h_max_temp, w_max_temp))
                mask_temp, x, y, h, w = mask_u2n, x_u2n, y_u2n, h_u2n, w_u2n

        for ind, file in enumerate(file_list[:data_len]):

            if self.debug: print(f"{self.mode} file:", file)

            img_temp = scipy.misc.imread(os.path.join(self.root, self.base_folder, file))


            ## Downscale large images for memory efficiency
            if self.ds_name != "CUB":

                img_temp = (img_temp).astype(np.uint8)

                if (img_temp.shape[0] > self.max_res) or (img_temp.shape[1] > self.max_res):

                    if self.debug and ind < 10:
                        print("Before:", img_temp.shape[0], img_temp.shape[1])
                        img_name = ("test/img_before_tr" + str(ind) + ".png")
                        Image.fromarray(img_temp, mode='RGB').save(img_name)

                    if img_temp.shape[0] > img_temp.shape[1]:
                        downscale_coef = img_temp.shape[0] / self.max_res 
                    else:
                        downscale_coef = img_temp.shape[1] / self.max_res 
                    
                    img_temp = transform_sk.resize( img_temp, ( int((img_temp.shape[0] // downscale_coef)), int((img_temp.shape[1] // downscale_coef)) ), \
                                                                mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True )
                
                    if self.debug and ind < 10:
                        print("After:", img_temp.shape[0], img_temp.shape[1])
                        img_temp = (img_temp).astype(np.uint8)
                        img_name = ("test/img_after_tr" + str(ind) + ".png")
                        Image.fromarray(img_temp, mode='RGB').save(img_name)
                else:
                    if self.debug and ind < 10:
                        print("Normal:", img_temp.shape[0], img_temp.shape[1])
                        img_name = ("test/img_normal_tr" + str(ind) + ".png")
                        Image.fromarray(img_temp, mode='RGB').save(img_name)

            h_max = img_temp.shape[0] # y
            w_max = img_temp.shape[1] # x
            #ch_max = img_temp.shape[2] # ch

            if self.gt_bbox:
                x, y, w, h = bb_list[ind] # x - distance from top up left (width), y - distance from top up left (height)
            
            if self.gt_parts:
                parts = parts_list[ind] # list of 15 parts with [x, y] center corrdinates

                #mask_temp = np.zeros((int(h_max), int(w_max))) # Black mask
                mask_temp = np.ones((int(h_max), int(w_max)))

                p_part = 16*3 # padding around center point

                for part_n in range(len(parts)):
                    part = parts[part_n]

                    if part[0] != 0:
                        x_min_p = part[1] - p_part
                        if x_min_p < 0:
                            x_min_p = 0
                        x_max_p = part[1] + p_part
                        if x_max_p > w_max:
                            x_max_p = w_max

                        y_min_p = part[2] - p_part
                        if y_min_p < 0:
                            y_min_p = 0
                        y_max_p = part[2] + p_part
                        if y_max_p > h_max:
                            y_max_p = h_max

                        #mask_temp[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 1 # Black mask
                        mask_temp[int(y_min_p):int(y_max_p), int(x_min_p):int(x_max_p)] = 0

            if self.sm_vit and self.full_ds:
                mask_temp = mask_u2n_list[ind]
                x = x_u2n_list[ind]
                y = y_u2n_list[ind]
                h = h_u2n_list[ind]
                w = w_u2n_list[ind]


            ## Image and Mask Padding:
            if self.sm_vit or self.gt_bbox:
                if self.padding:
                    p = 15 # extra space around bbox
                else:
                    p = 0

                x_min = x - p 
                if x_min < 0:
                    x_min = 0
                x_max = x + w + p
                if x_max > w_max:
                    x_max = w_max

                y_min = y - p
                if y_min < 0:
                    y_min = 0
                y_max = y + h + p
                if y_max > h_max:
                    y_max = h_max

                if h_max <=1:
                    print("[WARNING] bad_h", h_max)
                if w_max <=1:
                    print("[WARNING] bad_w", w_max)
                if y_min >= y_max:
                    print("[WARNING] bad_y", "min:", y_min, "max:", y_max)
                    print("[WARNING] y:", y, "h:", h)
                if x_min >= x_max:
                    print("[WARNING] bad_x", "min:", x_min, "max:", x_max)
                    print("[WARNING] x:", x, "w:", w)                                  
            ##


            ## Crop with bbox:
            if self.rand_crop:
                #prob_rcrop = 0.25 # 0.07 # 0.3 # 0.5
                #rand_crop_mask_temp = bool(random.random() < prob_rcrop)
                #if rand_crop_mask_temp:

                h_max_img = img_temp.shape[0]
                w_max_img = img_temp.shape[1]

                #h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                h_crop_mid_img = int(h_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

                #w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
                w_crop_mid_img = int(w_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img

                # Crop image with bbox:
                if len(img_temp.shape) == 3:
                    img_temp = img_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    img_temp = img_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                # Crop mask with bbox:
                mask_temp = mask_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]

            else:
                # Crop image with bbox:
                if len(img_temp.shape) == 3:
                    if self.gt_parts:
                        for j in range(3):
                            img_temp[:, :, j] = img_temp[:, :, j] * mask_temp # Black mask
                    else:
                        #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                        img_temp = img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
                else:
                    if self.gt_parts:                        
                        img_temp[:, :] = img_temp[:, :] * mask_temp # Black mask:
                    else:
                        img_temp = img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w

                # Crop mask with bbox:
                if self.sm_vit or self.gt_bbox:
                    mask_temp = mask_temp[int(y_min):int(y_max), int(x_min):int(x_max)]
            ##


            if ( (img_temp.shape[0] != mask_temp.shape[0]) or (img_temp.shape[1] != mask_temp.shape[1]) ):
                print("[WARNING] Image shape does not match mask shape for sample:", ind, ". \t" , "Found shapes:", img_temp.shape, mask_temp.shape)

            img.append(img_temp)
            mask.append(mask_temp)

        return img, mask


    def generic_preprocess_lowMem(self, file_list, file_list_full, shape_hw_list):

        print(f'[INFO] Pre-processing {self.mode} files in the low memory mode...')

        if self.sm_vit:
            if self.full_ds:
                mask_u2n_list, x_u2n_list, y_u2n_list, h_u2n_list, w_u2n_list = \
                    mask_hw(full_ds=self.full_ds, img_path=file_list_full, shape_hw=shape_hw_list)
            else: # for debug
                img_path = os.path.join(self.root, self.base_folder, file_list)
                img_temp = scipy.misc.imread(img_path)
                h_max_temp = img_temp.shape[0] # y
                w_max_temp = img_temp.shape[1] # x
                mask_u2n, x_u2n, y_u2n, h_u2n, w_u2n = \
                    mask_hw(full_ds=self.full_ds, img_path=img_path, shape_hw=(h_max_temp, w_max_temp))
                mask_temp, x, y, h, w = mask_u2n, x_u2n, y_u2n, h_u2n, w_u2n
                # mask_u2n_list, x_u2n_list, y_u2n_list, h_u2n_list, w_u2n_list = mask_temp, x, y, h, w

        return mask_u2n_list, x_u2n_list, y_u2n_list, h_u2n_list, w_u2n_list


    def generic_getitem(self, index,  img, mask):

        if self.is_train:
            if self.rand_crop_im_mask:
                h_max_img = img.shape[0]
                w_max_img = img.shape[1]

                h_crop_mid_img = int(h_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)
                w_crop_mid_img = int(w_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

                h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
                w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

                h_crop_max_img = h_crop_mid_img + h_crop_min_img
                w_crop_max_img = w_crop_mid_img + w_crop_min_img

                # Crop image:
                if len(img.shape) == 3:
                    img = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
                else:
                    img = img[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

                # Crop mask:
                mask = mask[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        if self.ds_name != "CUB":
            img = (img).astype(np.uint8)

        img = Image.fromarray(img, mode='RGB')

        if self.debug and index < 10:
            img_tem = transforms.ToTensor()(img)
            img_name = ("test/img_bef" + str(index) + ".png")
            save_image( img_tem, img_name)
        

        ## Image:
        if self.transform is not None:
            if self.is_train:
                if not self.flip_mask_as_image: # normal
                    img = self.transform(img) 
                else:
                    if random.random() < 0.5:
                        flipped = False
                        img = self.transform(img)
                    else: 
                        flipped = True
                        transform_img_flip = transforms.Compose([
                            #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                            #transforms.RandomCrop((args.img_size, args.img_size)),
                        
                            transforms.Resize((self.img_size, self.img_size),Image.BILINEAR), # my for bbox

                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                            transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!

                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
                        img = transform_img_flip(img)
            else:
                img = self.transform(img)

        if self.debug and index < 10:
            img_name = ("test/img_aft" + str(index) + ".png")
            save_image( img, img_name)            


        ## Mask:
        if self.crop_mask:
            h_max_im = mask.shape[0]
            w_max_im = mask.shape[1]

            h_crop_mid = int(h_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)
            w_crop_mid = int(w_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

            cropped = np.ones_like(mask)

            if self.mid_val:
                cropped = cropped * 0.125 # (for 0.2)

            h_crop_min = random.randint(0, (h_max_im - h_crop_mid)) # 40) #, 400-360) #, h - th)
            w_crop_min = random.randint(0, (w_max_im - w_crop_mid)) # 40)  #, 400-360) #, w - tw)

            h_crop_max = h_crop_mid + h_crop_min
            w_crop_max = w_crop_mid + w_crop_min

            cropped[int(h_crop_min):int(h_crop_max), int(w_crop_min):int(w_crop_max)] = 0
            
            mask = mask + cropped

            if self.mid_val:
                mask[mask > 1.1] = 1
            else:
                mask[mask > 1] = 1

        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')            

        if self.debug and index < 10:
            mask_tem = transforms.ToTensor()(mask)
            img_name = ("test/mask_bef" + str(index) + ".png")
            save_image( mask_tem, img_name)


        mask_size = int(self.img_size // 16)

        if self.is_train:
            if not self.flip_mask_as_image: # normal
                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(), #(mode='1'),

                    # non-overlapped:
                    transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    transforms.ToTensor()
                    ])
            else:
                if flipped:
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),
                        transforms.RandomHorizontalFlip(p=1.0),
                                                                        
                        # non-overlapped:
                        transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        transforms.ToTensor()
                        ])
                else:
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),
                                        
                        # non-overlapped:
                        transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        transforms.ToTensor()
                        ])
        else:
            transform_mask = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(), #(mode='1'),
                                 
                # non-overlapped:
                transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                transforms.ToTensor()])

        mask = transform_mask(mask)

        if self.debug and index < 10:
            img_name = ("test/mask_aft" + str(index) + ".png")
            save_image(mask, img_name)

        mask = torch.flatten(mask)

        return img, mask


    def generic_getitem_lowMem(self, index):

        file_temp = self.file_list[index]
        img_temp = scipy.misc.imread(os.path.join(self.root, self.base_folder, file_temp))

        ## Downscale large images for memory efficiency
        if self.ds_name != "CUB":

            self.gt_bbox = False
            self.gt_parts = False

            img_temp = (img_temp).astype(np.uint8)

            if (img_temp.shape[0] > self.max_res) or (img_temp.shape[1] > self.max_res):

                if self.debug and index < 10:
                    print("Before:", img_temp.shape[0], img_temp.shape[1])
                    img_name = ("test/img_before_tr" + str(index) + ".png")
                    Image.fromarray(img_temp, mode='RGB').save(img_name)

                if img_temp.shape[0] > img_temp.shape[1]:
                    downscale_coef = img_temp.shape[0] / self.max_res 
                else:
                    downscale_coef = img_temp.shape[1] / self.max_res 
                
                img_temp = transform_sk.resize( img_temp, ( int((img_temp.shape[0] // downscale_coef)), int((img_temp.shape[1] // downscale_coef)) ), \
                                                            mode='constant', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True )
            
                if self.debug and index < 10:
                    print("After:", img_temp.shape[0], img_temp.shape[1])
                    img_temp = (img_temp).astype(np.uint8)
                    img_name = ("test/img_after_tr" + str(index) + ".png")
                    Image.fromarray(img_temp, mode='RGB').save(img_name)
            else:
                if self.debug and index < 10:
                    print("Normal:", img_temp.shape[0], img_temp.shape[1])
                    img_name = ("test/img_normal_tr" + str(index) + ".png")
                    Image.fromarray(img_temp, mode='RGB').save(img_name)        
        ##


        h_max = img_temp.shape[0] # y
        w_max = img_temp.shape[1] # x
        #ch_max = img_temp.shape[2] # ch

        mask_temp = self.mask_u2n_list[index]
        x, y, h, w = self.x_u2n_list[index], self.y_u2n_list[index], self.h_u2n_list[index], self.w_u2n_list[index]


        ## Image and Mask Padding:
        if self.sm_vit or self.gt_bbox:
            if self.padding:
                p = 15 # extra space around bbox
            else:
                p = 0

            x_min = x - p 
            if x_min < 0:
                x_min = 0
            x_max = x + w + p
            if x_max > w_max:
                x_max = w_max

            y_min = y - p
            if y_min < 0:
                y_min = 0
            y_max = y + h + p
            if y_max > h_max:
                y_max = h_max

            if h_max <=1:
                print("[WARNING] bad_h", h_max)
            if w_max <=1:
                print("[WARNING] bad_w", w_max)
            if y_min >= y_max:
                print("[WARNING] bad_y", "min:", y_min, "max:", y_max)
                print("[WARNING] y:", y, "h:", h)
            if x_min >= x_max:
                print("[WARNING] bad_x", "min:", x_min, "max:", x_max)
                print("[WARNING] x:", x, "w:", w)                                  
        ##


        ## Crop with bbox:
        if self.rand_crop:
            #prob_rcrop = 0.25 # 0.07 # 0.3 # 0.5
            #rand_crop_mask_temp = bool(random.random() < prob_rcrop)
            #if rand_crop_mask_temp:

            h_max_img = img_temp.shape[0]
            w_max_img = img_temp.shape[1]

            #h_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
            h_crop_mid_img = int(h_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

            #w_crop_mid = 368 # 384 (92%), 368 (84%), 352 (77%), 336 (70%), 320 (64%), 304 (57%)
            w_crop_mid_img = int(w_max_img * 0.88) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

            h_crop_min_img = random.randint(0, (h_max_img - h_crop_mid_img)) # 40) #, 400-360) #, h - th)
            w_crop_min_img = random.randint(0, (w_max_img - w_crop_mid_img)) # 40)  #, 400-360) #, w - tw)

            h_crop_max_img = h_crop_mid_img + h_crop_min_img
            w_crop_max_img = w_crop_mid_img + w_crop_min_img

            # Crop image with bbox:
            if len(img_temp.shape) == 3:
                img_temp = img_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img), :] # h, w, ch
            else:
                img_temp = img_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)] # h, w

            # Crop mask with bbox:
            mask_temp = mask_temp[int(h_crop_min_img):int(h_crop_max_img), int(w_crop_min_img):int(w_crop_max_img)]

        else:
            # Crop image with bbox:
            if len(img_temp.shape) == 3:
                if self.gt_parts:
                    for j in range(3):
                        img_temp[:, :, j] = img_temp[:, :, j] * mask_temp # Black mask
                else:
                    #test_img_temp = test_img_temp[int(y):int(y + h), int(x):int(x + w), :] # h, w, ch
                    img_temp = img_temp[int(y_min):int(y_max), int(x_min):int(x_max), :] # h, w, ch
            else:
                if self.gt_parts:                        
                    img_temp[:, :] = img_temp[:, :] * mask_temp # Black mask:
                else:
                    img_temp = img_temp[int(y_min):int(y_max), int(x_min):int(x_max)] # h, w

            # Crop mask with bbox:
            if self.sm_vit or self.gt_bbox:
                mask_temp = mask_temp[int(y_min):int(y_max), int(x_min):int(x_max)]
        ##


        if ( (img_temp.shape[0] != mask_temp.shape[0]) or (img_temp.shape[1] != mask_temp.shape[1]) ):
            print("[WARNING] Image shape does not match mask shape for sample:", index, ". \t" , \
                        "Found shapes:", img_temp.shape, mask_temp.shape)


        img = img_temp

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        if self.ds_name != "CUB":
            img = (img).astype(np.uint8)

        img = Image.fromarray(img, mode='RGB')

        if self.debug and index < 10:
            img_tem = transforms.ToTensor()(img)
            img_name = ("test/img_bef" + str(index) + ".png")
            save_image( img_tem, img_name)
        

        ## Image:
        if self.transform is not None:
            if self.is_train:
                if not self.flip_mask_as_image: # normal
                    img = self.transform(img) 
                else:
                    if random.random() < 0.5:
                        flipped = False
                        img = self.transform(img)
                    else: 
                        flipped = True
                        transform_img_flip = transforms.Compose([
                            #transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                            #transforms.RandomCrop((args.img_size, args.img_size)),
                        
                            transforms.Resize((self.img_size, self.img_size),Image.BILINEAR), # my for bbox

                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (FFVT)
                            transforms.RandomHorizontalFlip(p=1.0), # !!! FLIPPING in dataset.py !!!

                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
                        img = transform_img_flip(img)
            else:
                img = self.transform(img)

        if self.debug and index < 10:
            img_name = ("test/img_aft" + str(index) + ".png")
            save_image( img, img_name)            


        ## Mask:
        mask = mask_temp

        if self.crop_mask:
            h_max_im = mask.shape[0]
            w_max_im = mask.shape[1]

            h_crop_mid = int(h_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)
            w_crop_mid = int(w_max_im * 0.84) # 384 (92% - 0.96), 368 (84% - 0.92), 352 (77% - 0.88), 336 (70% - 0.84), 320 (64% - 0.80), 304 (57% - 0.75)

            cropped = np.ones_like(mask)

            if self.mid_val:
                cropped = cropped * 0.125 # (for 0.2)

            h_crop_min = random.randint(0, (h_max_im - h_crop_mid)) # 40) #, 400-360) #, h - th)
            w_crop_min = random.randint(0, (w_max_im - w_crop_mid)) # 40)  #, 400-360) #, w - tw)

            h_crop_max = h_crop_mid + h_crop_min
            w_crop_max = w_crop_mid + w_crop_min

            cropped[int(h_crop_min):int(h_crop_max), int(w_crop_min):int(w_crop_max)] = 0
            
            mask = mask + cropped

            if self.mid_val:
                mask[mask > 1.1] = 1
            else:
                mask[mask > 1] = 1

        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')            

        if self.debug and index < 10:
            mask_tem = transforms.ToTensor()(mask)
            img_name = ("test/mask_bef" + str(index) + ".png")
            save_image( mask_tem, img_name)


        mask_size = int(self.img_size // 16)

        if self.is_train:
            if not self.flip_mask_as_image: # normal
                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(), #(mode='1'),

                    # non-overlapped:
                    transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    transforms.ToTensor()
                    ])
            else:
                if flipped:
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),
                        transforms.RandomHorizontalFlip(p=1.0),
                                                                        
                        # non-overlapped:
                        transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        transforms.ToTensor()
                        ])
                else:
                    transform_mask = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.ToPILImage(), #(mode='1'),
                                        
                        # non-overlapped:
                        transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                        transforms.ToTensor()
                        ])
        else:
            transform_mask = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(), #(mode='1'),
                                 
                # non-overlapped:
                transforms.Resize((mask_size, mask_size), interpolation=Image.NEAREST), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                transforms.ToTensor()])

        mask = transform_mask(mask)

        if self.debug and index < 10:
            img_name = ("test/mask_aft" + str(index) + ".png")
            save_image(mask, img_name)

        mask = torch.flatten(mask)

        return img, mask




class CUB(Generic_smvit_DS):

    def __init__(self, ds_name, root, is_train=True, data_len=None, transform=None, sm_vit=True, low_memory=True, img_size=400):

        self.ds_name = ds_name
        self.img_size = img_size
        self.max_res = int(self.img_size * 1.5)


        self.full_ds = True # pre-processing full dataset
        self.padding = True # image and mask padding
        self.rand_crop = False # if no other cropping

        self.flip_mask_as_image = True # if False - turn on RandomHorizontalFlip in data_utils !!!
        self.rand_crop_im_mask = False # randomly crop both image and mask

        self.crop_mask = False # randomly crop mask only
        self.mid_val = False # 3-state mask

        self.debug = False # for debug info
        if self.debug:
            os.makedirs("./test")

        self.gt_bbox = False # for other experiments
        self.gt_parts = False # for other experiments


        self.sm_vit = sm_vit
        self.low_memory = low_memory

        if (self.sm_vit + self.gt_bbox + self.gt_parts) > 1 :
            raise Exception("Only one cropping mode (SM-ViT, bbox, parts) can be chosen")


        self.root = root
        self.base_folder = "images"
        self.transform = transform

        self.is_train = is_train

        if self.is_train:
            self.mode = "Train"
        else:
            self.mode = "Test"


        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        if self.is_train:
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            file_list_full = [ os.path.join(self.root, self.base_folder, x)  for i, x in zip(train_test_list, img_name_list) if i]
        else:
            self.file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
            file_list_full = [ os.path.join(self.root, self.base_folder, x)  for i, x in zip(train_test_list, img_name_list) if not i]


        if self.sm_vit:
            print(f"[INFO] Preparing {self.mode} shape_hw list...")

            shape_hw_list = []

            for img_name in self.file_list:
                img_temp = scipy.misc.imread(os.path.join(self.root, self.base_folder, img_name))
                shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)
                shape_hw_list.append(shape_hw_temp)

            if self.low_memory:
                self.mask_u2n_list, self.x_u2n_list, self.y_u2n_list, self.h_u2n_list, self.w_u2n_list = \
                        super(CUB, self).generic_preprocess_lowMem(self.file_list,
                                                                    file_list_full, 
                                                                    shape_hw_list
                                                                    )
                del shape_hw_list
                del file_list_full
                gc.collect()                                                          
            else:
                self.img, self.mask = \
                    super(CUB, self).generic_preprocess(self.file_list, 
                                                        file_list_full, 
                                                        shape_hw_list,
                                                        data_len,
                                                        train_test_list
                                                        )
        else:
            self.img = \
                [scipy.misc.imread(os.path.join(self.root, self.base_folder, file)) \
                    for file in self.file_list[:data_len]]

        if self.is_train:
            self.label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        else:
            self.label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

        self.imgname = [x for x in self.file_list[:data_len]]



    def __getitem__(self, index):

        if self.sm_vit:
            if self.low_memory:
                target, imgname = self.label[index], self.imgname[index]
                img, mask = super(CUB, self).generic_getitem_lowMem(index)
            else:
                img, target, imgname, mask = self.img[index], self.label[index], self.imgname[index], self.mask[index]
                img, mask = super(CUB, self).generic_getitem(index, img, mask)
            
            return img, target, mask

        else:
            img, target, imgname = self.img[index], self.label[index], self.imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)

            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img, target


    def __len__(self):
        return len(self.label)




class dogs(Generic_smvit_DS): #(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'dog'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 ds_name,
                 root,
                 is_train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False,
                 sm_vit=True,
                 low_memory=True,
                 img_size=400):


        self.ds_name = ds_name
        self.img_size = img_size
        self.max_res = int(self.img_size * 1.5)


        self.full_ds = True # pre-processing full dataset
        self.padding = True # image and mask padding
        self.rand_crop = False # if no other cropping
    
        self.flip_mask_as_image = True # if False - turn on RandomHorizontalFlip in data_utils !!!
        self.rand_crop_im_mask = False # randomly crop both image and mask

        self.crop_mask = False # randomly crop mask only
        self.mid_val = False # 3-state mask

        self.debug = False # for debug info
        if self.debug:
            os.makedirs("./test")


        self.sm_vit = sm_vit
        self.low_memory = low_memory

        # self.root = join(os.path.expanduser(root), self.folder)
        self.root = root
        self.base_folder = "Images"

        self.is_train = is_train

        if self.is_train:
            self.mode = "Train"
        else:
            self.mode = "Test"

        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                        for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]


        data_len = None

        if self.sm_vit:
            print(f"[INFO] Preparing {self.mode} shape_hw list...")

            shape_hw_list = []
            self.file_list = []
            file_list_full = []

            for image_name, target_class in self._flat_breed_images:
                img_name = join(self.images_folder, image_name)
                img_temp = scipy.misc.imread(os.path.join(img_name))
                shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)

                if (shape_hw_temp[0] > self.max_res) or (shape_hw_temp[1] > self.max_res):
                    if shape_hw_temp[0] > shape_hw_temp[1]:
                        downscale_coef = shape_hw_temp[0] / self.max_res
                    else:
                        downscale_coef = shape_hw_temp[1] / self.max_res
                        
                    shape_hw_temp[0] = int(shape_hw_temp[0] // downscale_coef)
                    shape_hw_temp[1] = int(shape_hw_temp[1] // downscale_coef)

                shape_hw_list.append(shape_hw_temp)
                self.file_list.append(image_name)
                file_list_full.append(img_name)

            if self.low_memory:
                self.mask_u2n_list, self.x_u2n_list, self.y_u2n_list, self.h_u2n_list, self.w_u2n_list = \
                        super(dogs, self).generic_preprocess_lowMem(self.file_list,
                                                                    file_list_full, 
                                                                    shape_hw_list
                                                                    )
                del shape_hw_list
                del file_list_full
                gc.collect()                                                          
            else:
                self.img, self.mask = \
                    super(dogs, self).generic_preprocess(self.file_list, 
                                                        file_list_full, 
                                                        shape_hw_list,
                                                        data_len
                                                        )            

        if self.is_train:
            self.label = [x for i, x in self._flat_breed_images][:data_len]
        else:
            self.label = [x for i, x in self._flat_breed_images][:data_len]

        self.imgname = [x for x in self.file_list[:data_len]]                  


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """

        if self.sm_vit:
            if self.low_memory:
                target, imgname = self.label[index], self.imgname[index]
                img, mask = super(dogs, self).generic_getitem_lowMem(index)
            else:
                img, target, imgname, mask = self.img[index], self.label[index], self.imgname[index], self.mask[index]
                img, mask = super(dogs, self).generic_getitem(index, img, mask)
            
            return img, target, mask

        else:
            image_name, target = self._flat_breed_images[index]
            image_path = join(self.images_folder, image_name)
            img = Image.open(image_path).convert('RGB')

            if self.cropped:
                img = img.crop(self._flat_breed_annotations[index][1])

            if self.transform:
                img = self.transform(img)

            if self.target_transform:
                target = self.target_transform(target)

            return img, target


    def __len__(self):
        return len(self._flat_breed_images)


    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))


    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes


    def load_split(self):
        if self.is_train:
            # split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            # labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
            split = scipy.io.loadmat(join(self.root, 'splits/train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'splits/train_list.mat'))['labels']
        else:
            # split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            # labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']
            split = scipy.io.loadmat(join(self.root, 'splits/test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'splits/test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))


    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts




class NABirds(Generic_smvit_DS): #(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    #base_folder = 'nabirds/images'

    def __init__(self, ds_name, root, is_train=True, data_len=None, transform=None, sm_vit=True, low_memory=True, img_size=448):

        self.ds_name = ds_name
        self.img_size = img_size
        self.max_res = int(self.img_size * 1.25) # 1.5


        self.full_ds = True # pre-processing full dataset
        self.padding = True # image and mask padding
        self.rand_crop = False # if no other cropping

        self.flip_mask_as_image = True # if False - turn on RandomHorizontalFlip in data_utils !!!
        self.rand_crop_im_mask = False # randomly crop both image and mask

        self.crop_mask = False # randomly crop mask only
        self.mid_val = False # 3-state mask

        self.debug = False # for debug info
        if self.debug:
            os.makedirs("./test")


        self.sm_vit = sm_vit
        self.low_memory = low_memory

        dataset_path = os.path.join(root)
        self.root = root
        self.base_folder = "images"

        self.loader = default_loader
        self.transform = transform

        self.is_train = is_train

        if self.is_train:
            self.mode = "Train"
        else:
            self.mode = "Test"


        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.is_train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)


        self.data_len = None

        if self.sm_vit:
            print(f"[INFO] Preparing {self.mode} shape_hw list...")
            
            shape_hw_list = []
            self.file_list = []
            file_list_full = []

            for sample in self.data.iloc:
                image_name = sample.filepath
                img_name_full = join(self.root, self.base_folder, image_name)
                img_temp = scipy.misc.imread(os.path.join(img_name_full))
                shape_hw_temp = [img_temp.shape[0], img_temp.shape[1]] # h_max (y), w_max (x)

                if (shape_hw_temp[0] > self.max_res) or (shape_hw_temp[1] > self.max_res):
                    if shape_hw_temp[0] > shape_hw_temp[1]:
                        downscale_coef = shape_hw_temp[0] / self.max_res
                    else:
                        downscale_coef = shape_hw_temp[1] / self.max_res
                        
                    shape_hw_temp[0] = int(shape_hw_temp[0] // downscale_coef)
                    shape_hw_temp[1] = int(shape_hw_temp[1] // downscale_coef)

                shape_hw_list.append(shape_hw_temp)
                self.file_list.append(image_name)
                file_list_full.append(img_name_full)

            if self.low_memory:
                self.mask_u2n_list, self.x_u2n_list, self.y_u2n_list, self.h_u2n_list, self.w_u2n_list = \
                        super(NABirds, self).generic_preprocess_lowMem(self.file_list,
                                                                    file_list_full, 
                                                                    shape_hw_list
                                                                    )
                del shape_hw_list
                del file_list_full
                gc.collect()                                                          
            else:
                self.img, self.mask = \
                    super(NABirds, self).generic_preprocess(self.file_list, 
                                                        file_list_full, 
                                                        shape_hw_list,
                                                        self.data_len
                                                        )

        if self.is_train:
            self.label = [ (self.label_map[x.target]) for x in self.data.iloc ][:self.data_len]
        else:
            self.label = [ (self.label_map[x.target]) for x in self.data.iloc ][:self.data_len]

        self.imgname = [x for x in self.file_list[:self.data_len]]



    def __getitem__(self, index):

        if self.sm_vit:
            if self.low_memory:
                target, imgname = self.label[index], self.imgname[index]
                img, mask = super(NABirds, self).generic_getitem_lowMem(index)
            else:
                img, target, imgname, mask = self.img[index], self.label[index], self.imgname[index], self.mask[index]
                img, mask = super(NABirds, self).generic_getitem(index, img, mask)
            
            return img, target, mask

        else:
            sample = self.data.iloc[index]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = self.label_map[sample.target]

            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)

            return img, target


    def __len__(self):
        return len(self.data)


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])
    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents





### Not optimised datasets:

class INat2017(VisionDataset):
    """`iNaturalist 2017 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(INat2017, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            if not (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['annos'][1]))):
                print('Downloading...')
                self._download()
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.file_list['imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['annos'][1]))
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _download(self):
        for url, filename, md5 in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
            if not check_integrity(os.path.join(self.root, filename), md5):
                raise RuntimeError("File not found or corrupted.")



class CarsDataset(Dataset):

    def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)
            for c in self.car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[idx][-2][0][0]
        car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        assert car_class < 196
        
        if self.transform:
            image = self.transform(image)

        # return image, car_class, img_name
        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()

def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images
    
def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)



class FGVC_aircraft():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = scipy.misc.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        else:
            img, target = scipy.misc.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)
