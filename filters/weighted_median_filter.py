#!/usr/bin/python3
import getopt
import queue
import sys
import os

import cv2


# Clase util para representar las celdas y sus propiedades
class Coor:
    def __init__(self, x, y, weight, distance):
        self.x = x
        self.y = y
        self.weight = weight
        self.distance = distance

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_weight(self):
        return self.weight

    def get_distance(self):
        return self.distance


def gen_kernel_mask(window_size):
    kernel_mask = []

    # Initializamos el kernel
    for _ in range(window_size):
        kernel_mask.append([-1.0 for _ in range(window_size)])

    # Calculamos el centro del Kernel
    kernel_center_x, kernel_center_y = window_size // 2, window_size // 2

    # Initializamos el centro del kernel con valor 1.0 (el peso maximo)
    kernel_center_weight = 1.0

    # La distance del centro es obviamente 0
    kernel_center_distance = 0.0

    # Valores auxiliares que nos ayudaran para tomar las coordenadas que rodean la celda en cuestion
    a_x = [0, 1, 1, 1, 0, -1, -1, -1]
    a_y = [-1, -1, 0, 1, 1, 1, 0, -1]

    # Calculamos el peso a restar por distancia (el peso puede ser arbitrario, estadistico, etc..)
    weight_per_distance = 1.0 / (float(window_size))

    # Creamos una cola
    q = queue.Queue()

    q.put(Coor(kernel_center_x, kernel_center_y, kernel_center_weight, kernel_center_distance))

    kernel_mask[kernel_center_x][kernel_center_y] = kernel_center_weight

    while not q.empty():
        # Celda actual
        cell = q.get()

        for i in range(len(a_x)):
            # Calculamos las coordenadas del celda vecina
            tg_cell_x, tg_cell_y = cell.get_x() + a_x[i], cell.get_y() + a_y[i]

            # Si nos salimos del ventana del kernel simplemente continuamos
            if tg_cell_x < 0 or tg_cell_x >= window_size or tg_cell_y < 0 or tg_cell_y >= window_size:
                continue

            # Si esta celda no tiene el valor por defecto significa que ya fue visitada, por lo tanto la ignoramos
            if kernel_mask[tg_cell_x][tg_cell_y] != -1.0:
                continue

            # Calculamos la distance del a celda vecina
            calculated_distance = cell.get_distance() + 1.0

            # Calculamos el valor del celda vecina
            # Vamos a mantener 0.05 como el peso minimo
            calculated_weight = max(1.0 - (calculated_distance * weight_per_distance), 0.05)

            # Agregamos la celda a la queue y realizamos el mismo proceso para esta celda
            neighbor_cell = Coor(tg_cell_x, tg_cell_y, calculated_weight, calculated_distance)
            q.put(neighbor_cell);

            # Guardamos el peso de la celda
            kernel_mask[tg_cell_x][tg_cell_y] = calculated_weight

    print("Kernel mask weight values:")
    for row in kernel_mask:
        print(row)
    return kernel_mask


def weighted_median_filter(image_name, window_size=3):
    res_dir = os.environ["PY_IMG"]
    if res_dir is None:
        print("[ERROR] Resources path isn't defined")

    # Convertimos a escala de grises
    original_image = cv2.imread(res_dir + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise Exception("[ERROR] Image not found in path.")

    print("Imagen: ", image_name)

    # Procuramos usar un tamaño impar
    if window_size % 2 == 0:
        window_size += 1

    # Dimensiones de la imagen
    width, height = original_image.shape
    print("Dimensiones: ", width, height)

    # Kernel/window radio size
    radio_size = window_size // 2

    kernel_mask = gen_kernel_mask(window_size)
    # Image de salida
    output_image = original_image.copy()

    for x in range(radio_size, width):
        for y in range(radio_size, height):
            # Guarda el peso y el valor del pixel
            pixel_values = []

            # Ciclo principal de la nuestra ventana
            for i in range(-radio_size, radio_size + 1):
                for j in range(-radio_size, radio_size + 1):
                    # Celdas (x, y) a considerar
                    tg_x, tg_y = x + i, y + j

                    # Si nos salidos de los bordes omitimos
                    if tg_x < 0 or tg_x >= width or tg_y < 0 or tg_y >= height:
                        continue

                    # Calculamos las coordenadas (row, col) para el kernel mask
                    k_x, k_y = i + radio_size, j + radio_size

                    # Calculamos el peso en base a la multiplicacion con el kernel
                    weight_value = kernel_mask[k_x][k_y] * float(original_image[tg_x][tg_y])

                    # Guardamos el peso y valor del pixel en cuestion
                    pixel_values.append([weight_value, original_image[tg_x][tg_y]])

            # ordenamos la lista, toma como principal prioridad el peso del pixel y como segunda prioridad toma el valor del pixel
            pixel_values.sort()

            # La median es el valor que se encuentra en medio de un conjunto de datos previamente ordenados
            # El valor de nuestro pixel se encuentra en la segunda posicion, osea 1
            median = round(pixel_values[len(pixel_values) // 2][1])
            output_image[x, y] = median

    cv2.imshow("Original image (weighted median filter)", original_image)
    cv2.imshow("Output image (weighted median filter)", output_image)

    print("\nPresione ENTER para continuar.\n")
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
        weighted_median_filter(image, window_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[Exception] ", str(e))
