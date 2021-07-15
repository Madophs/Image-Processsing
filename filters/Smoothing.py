#!/usr/bin/python3
import os
import cv2

def Smoothing(image_name):
    res_dir = os.environ["PY_IMG"]
    if res_dir is None:
        print("[ERROR] Resources path isn't defined")

    # Convertimos a escala de grises
    original_image = cv2.imread(res_dir + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise Exception("[ERROR] Image not found in path.")

    img_width, img_height = original_image.shape
    output_image = original_image.copy()

    kernel_size = 3
    kernel_radio = kernel_size // 2
    kernel_window =\
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]

    for x in range(img_width):
        for y in range(img_height):
            items_sum, pixel_sum = 0, 0
            for i in range(-kernel_radio, kernel_radio+1):
                for j in range(-kernel_radio, kernel_radio+1):
                    # Target coordinates for image
                    tg_x, tg_y = i + x, j + y

                    # Target coordinates for kernel
                    ktg_x, ktg_y = i + kernel_radio, j + kernel_radio

                    # if out of bounds continue
                    if tg_x < 0 or tg_x >= img_width or tg_y < 0 or tg_y >= img_height:
                        continue

                    pixel_sum += kernel_window[ktg_x][ktg_y] * original_image[tg_x, tg_y]
                    items_sum += kernel_window[ktg_x][ktg_y]

            new_pixel = abs(pixel_sum // items_sum)
            output_image[x, y] = new_pixel

    print("Kernel window", kernel_window)
    cv2.imshow("Original image", original_image)
    cv2.imshow("Output image", output_image)
    cv2.waitKey(30000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Smoothing("noisy_1.jpeg")
    Smoothing("house.jpg")
