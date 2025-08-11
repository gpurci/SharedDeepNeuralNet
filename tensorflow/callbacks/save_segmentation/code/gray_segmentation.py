#!/usr/bin/python

import cv2
import numpy as np

def put_gray_segmentation_on_image(image, seg_image, down_size=1, alfa=0.7, beta=0.3):

    #print("\tout_image: size {}, max {}, min {}".format(seg_image.shape, seg_image.max(), seg_image.min()))
    input_shape = np.array(seg_image.shape[:2])*down_size
    input_shape = input_shape.astype(np.uint32)
    seg_image = cv2.resize(seg_image, (input_shape[1], input_shape[0]), interpolation = cv2.INTER_AREA)
    out_image = np.array((seg_image, seg_image, seg_image), dtype=np.float32)
    out_image = np.transpose(out_image, (1, 2, 0))
    
    image = image[:input_shape[0], :input_shape[1]]
    #print("\tout_image: size {}, max {}, min {}".format(out_image.shape, out_image.max(), out_image.min()))
    out_image *= beta
    image     = image.astype(np.float32)
    out_image = image * alfa + out_image
    out_image = out_image.astype(np.uint8)
    #print("\tout_image: size {}, max {}, min {}".format(out_image.shape, out_image.max(), out_image.min()))
    return out_image

def predict_cadran_gray_segmentation(model, image, split_crop=(32, 32), bite=10, down_size=1, batch_size=3):
    input_shape = np.array(image.shape[:2])//down_size
    out_image   = np.zeros(input_shape, dtype=np.uint8)
    down_split_crop = np.array(split_crop)//down_size
    for cadran_imgs, cadran_points in split_by_cadran(image, split_crop=split_crop, bite=bite):
        #print("size: cadran_imgs {}, cadran_points {}".format(len(cadran_imgs), len(cadran_points)))
        for crops, points in split_by_batch_size(cadran_imgs, cadran_points, batch_size=batch_size):
            tmp_batch_size = len(crops)
            crops    = np.array(crops)
            #print("\tcrops: size {}".format(crops.shape))
            predicts = model(crops, training=False).numpy()
            #print("\tpredicts: size {}".format(predicts.shape))
            predicts = np.array(predicts).reshape(tmp_batch_size, *down_split_crop)
            points   = np.array(points)//down_size
            #print("\tpredicts: size {}".format(predicts.shape))
            predicts = rescaling_gray(predicts)
            #print("\tpredicts: size {}".format(predicts.shape))
            put_in_image(out_image, predicts, points, split_crop=down_split_crop)

    #print("\tout_image: size {}, max {}, min {}".format(out_image.shape, out_image.max(), out_image.min()))
    return out_image

def rescaling_gray(pred_segments):
    for pred_img in pred_segments:
        valid_idx = np.argwhere(pred_img >= 0.5).T
        #print("\tvalid_idx: {}".format(valid_idx.shape))
        pred_img[:, :] = 0
        pred_img[valid_idx[0], valid_idx[1]] = 1
        #print("\trescaling: pred_img -> size {}, max {}, min {}".format(pred_img.shape, pred_img.max(), pred_img.min()))

    return pred_segments.astype(np.uint8)

def put_in_image(out_image, pred_segments, points, split_crop=(32, 32)):
    split_y_crop, split_x_crop = split_crop
    for pred_img, (st_y, st_x) in zip(pred_segments, points):
        tmp_img = out_image[st_y:st_y+split_y_crop, st_x:st_x+split_x_crop]
        tmp_img = np.bitwise_or(tmp_img, pred_img)
        #print("\t\tput_in_image: pred_img -> size {}, max {}, min {}".format(pred_img.shape, pred_img.max(), pred_img.min()))
        #print("\t\tput_in_image: tmp_img -> size {}, max {}, min {}".format(tmp_img.shape, tmp_img.max(), tmp_img.min()))
        out_image[st_y:st_y+split_y_crop, st_x:st_x+split_x_crop] = tmp_img

def split_by_batch_size(lst_images, lst_points, batch_size=3):
    for idx in range(0, len(lst_images), batch_size):
        crops  = lst_images[idx:idx+batch_size]
        points = lst_points[idx:idx+batch_size]
        yield crops, points

def split_by_cadran(image, split_crop=(32, 32), bite=10):
    height, width = image.shape[:2]

    for st_y in range(0, height, split_crop[0]-bite):
        end_y = min(height, st_y+split_crop[0])
        st_y  = max(0, end_y-split_crop[0])
        lst_crop = []
        points   = []
        for st_x in range(0, width, split_crop[1]):
            end_x = min(width, st_x+split_crop[1])
            st_x  = max(0, end_x-split_crop[1])
            crop = image[st_y:end_y, st_x:end_x]
            lst_crop.append(crop)
            points.append((st_y, st_x))
        yield lst_crop, points