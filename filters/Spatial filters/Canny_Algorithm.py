#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:46:49 2021

@author: ruizvilj
"""

import cv2 as cv
import os
import numpy as np
import math

class CannyFilter():
    
    def __init__(self, img_name):
        self.img_name = img_name
        self.assets = os.environ["PY_IMG"]
        self.original_img = None

    
    def edgeDetection(self):
        output_img = self.original_img.copy()
    
        img_width, img_height = self.original_img.shape
    
        # horizontal kernel mask
        kernel_x =\
            [
                [ 1, 2, 1],
                [ 0, 0, 0],
                [-1,-2,-1]
            ]
    
        # Vertical kernel mask
        kernel_y = \
            [
                [ 1, 0,-1],
                [ 2, 0,-2],
                [ 1, 0,-1]
            ]
    
        # Kernel dimensions
        kernel_size = 3
        kernel_radio = kernel_size // 2
    
        for x in range(img_width):
            for y in range(img_height):
                horizon_pixel_sum, vertical_pixel_sum = 0, 0
                for i in range(-kernel_radio, kernel_radio+1):
                    for j in range(-kernel_radio, kernel_radio+1):
                        # Target coordinate for original image
                        tg_x, tg_y = x + i, y + j
                        
                        # Target coordinates for kernel
                        ktg_x, ktg_y = i + kernel_radio, j + kernel_radio
    
                        if tg_x < 0 or tg_x >= img_width or tg_y < 0 or tg_y >= img_height:
                            continue
    
                        horizon_pixel_sum += kernel_x[ktg_x][ktg_y] * self.original_img[tg_x, tg_y]
                        vertical_pixel_sum += kernel_y[ktg_x][ktg_y] * self.original_img[tg_x, tg_y]
    
                # Formula to merge results = sqrt(img1_pixel ^ 2 + img2_pixel ^ 2)
                new_pixel = int(math.sqrt(pow(horizon_pixel_sum, 2) + pow(vertical_pixel_sum, 2)))
                output_img[x, y] = new_pixel
    
        return output_img
    
    def applyFilter(self):
        img_path = self.assets + "/" + self.img_name
        
        # Let's apply a grayscaling for easy manipulation
        self.original_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        output_img = np.copy(self.original_img)
        
        if self.original_img is None:
            raise Exception("[ERROR] Image not found in path.")
        
        # Let's remove some noisy using a Gausassian filter
        output_img = cv.GaussianBlur(self.original_img, (5, 5), 0)
        
        # Border detection
        output_img = self.edgeDetection()
        
        return output_img


cannyFilter = CannyFilter("house.jpg")
output_img = cannyFilter.applyFilter()

cv.imshow("Output image", output_img)

cv.waitKey(20000)
cv.destroyAllWindows()