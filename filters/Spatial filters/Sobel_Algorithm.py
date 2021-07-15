#!/usr/bin/python3
import cv2
import os
import math

def sobelFilterKernelRoutine(img_input, img_output, kernel_x, kernel_y, center=False):
    img_width, img_height = img_input.shape


def sobelFilter(img):
    res_dir = os.environ["PY_IMG"]
    if res_dir is None:
        raise Exception("[WARN] PY_IMG is undefined.")

    original_img = cv2.imread(res_dir+"/"+img, cv2.IMREAD_GRAYSCALE)
    output_img = original_img.copy()

    img_width, img_height = original_img.shape

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

                    horizon_pixel_sum += kernel_x[ktg_x][ktg_y] * original_img[tg_x, tg_y]
                    vertical_pixel_sum += kernel_y[ktg_x][ktg_y] * original_img[tg_x, tg_y]

            # Formula to merge results = sqrt(img1_pixel ^ 2 + img2_pixel ^ 2)
            new_pixel = int(math.sqrt(pow(horizon_pixel_sum, 2) + pow(vertical_pixel_sum, 2)))
            output_img[x, y] = new_pixel

    print(kernel_x, kernel_y)
    cv2.imshow("Original image", original_img)
    cv2.imshow("Border detection", output_img)
    cv2.waitKey(30000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sobelFilter("transit1.jpg")
    sobelFilter("house.jpg")
    sobelFilter("rombo_romboide.png")

