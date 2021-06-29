#!/usr/bin/python3
import getopt
import queue
import sys
import os

import cv2

def median_filter(image_name, window_size=3):
    res_dir = os.environ["PY_IMG"]
    if res_dir is None:
        print("[ERROR] Resources path isn't defined")

    # Convertimos a escala de grises
    original_image = cv2.imread(res_dir + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise Exception("[ERROR] Image not found in path.")

    # Dimensiones de la imagen
    width, height = original_image.shape
    print(width, height)

    radio_size = window_size // 2

    # Image de salida
    output_image = original_image.copy()

    for x in range(radio_size, width):
        for y in range(radio_size, height):
            pixel_arr = []
            # Ciclo principal de la nuestra ventana
            for i in range(-radio_size, radio_size + 1):
                for j in range(-radio_size, radio_size + 1):
                    # Celdas (x, y) a considerar
                    tg_x, tg_y = x + i, y + j
                    # Agregamos los pixeles que se encuentren en el rango de la imagen
                    if tg_x >= 0 and tg_x < width and tg_y >= 0 and tg_y < height:
                        # Guardamos el valor de cada pixel
                        pixel_arr.append(original_image[tg_x, tg_y])

            # ordenamos la lista
            pixel_arr.sort()

            # La median es el valor que se encuentra en medio de un conjunto de datos previamente ordenados
            median = int(round(pixel_arr[len(pixel_arr) // 2]))
            output_image[x, y] = median

    cv2.imshow("Original image (median filter)", original_image)
    cv2.imshow("Output image (median filter)", output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    opts, args = getopt.getopt(sys.argv[1:], "w:", "window-size=")
    window_size = 3
    for opt, arg in opts:
        if opt == "-w":
            window_size = int(arg)

    image_list = ["Prueba.tif", "Example_lena_denoise_noisy.jpg", "noisy_tiger.png"]
    for image in image_list:
        median_filter(image, window_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[Exception] ", str(e))
