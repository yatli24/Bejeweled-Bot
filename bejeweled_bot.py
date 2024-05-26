import pyautogui
from pyautogui import *
from PIL import ImageGrab
import cv2
import numpy as np
import pytesseract
import time
import sys
from PIL import Image
from skimage import io
import keyboard
import win32api, win32con

def get_average_color(image, x, y, cell_width, cell_height):

    area = image.crop((x, y, x + cell_width, y + cell_height))


    pixels = np.array(area)


    average_color = pixels.mean(axis=(0, 1))

    return tuple(average_color.astype(int))


def classify_color(rgb):

    color_thresholds = {
        'r': (143, 27, 43),
        'b': (35, 86, 133),
        'y': (118, 113, 52),
        'p': (101, 55, 112),
        'o': (139, 87, 48),
        'g': (35, 127, 59),
        'w': (125, 129, 132)
    }


    closest_color = min(color_thresholds,
                        key=lambda color: np.linalg.norm(np.array(rgb) - np.array(color_thresholds[color])))
    return closest_color


def create_color_matrix(image, rows, cols, cell_width, cell_height):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x = j * cell_width
            y = i * cell_height
            dominant_color = get_average_color(image, x, y, cell_width, cell_height)
            color_name = classify_color(dominant_color)
            row.append(color_name)
        matrix.append(row)
    return matrix


# Parameters


board_region = (367, 245, 340, 350)



rows, cols = 8, 8  # Adjust based on the game grid


image = pyautogui.screenshot(region=board_region)

cell_width = image.width // cols
cell_height = image.height // rows

color_matrix = create_color_matrix(image, rows, cols, cell_width, cell_height)

for row in color_matrix:
    formatted_row = ', '.join(row)
    print(f'[{formatted_row}]')


def match_color(a, b):
    if a == b:
        return True
    return False


def solve(matrix):


    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0


    for i in range(rows):
        for j in range(cols):
 
            current = matrix[i][j]
            current_left = matrix[i][j - 1] if j > 0 else None
            current_right = matrix[i][j + 1] if j < cols - 1 else None
            current_up = matrix[i - 1][j] if i > 0 else None
            current_down = matrix[i + 1][j] if i < rows - 1 else None

            if current_left is None and current_up is None:
                print(f'{current} -> Top left Corner')
            elif current_right is None and current_up is None:
                print(f'{current} -> Top Right Corner')
            elif current_left is None:
                print(f'{current} -> Left Edge')
            elif current_right is None:
                print(f'{current} -> Right Edge')
            elif current_up is None:
                print(f'{current} -> Top Edge')
            elif current_down is None:
                print(f'{current} -> Bottom Edge')


            print(
                f'Current: {current} Left: {current_left} Right: {current_right} Up: {current_up} Down: {current_down}')

    return


print(match_color(color_matrix[0][0], color_matrix[1][0]))

solve(color_matrix)

