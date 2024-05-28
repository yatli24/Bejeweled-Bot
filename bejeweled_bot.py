import pyautogui
from PIL import ImageGrab
import numpy as np
import time
from PIL import Image

# Helpers for gem recognition

def get_average_color(image, x, y, cell_width, cell_height):
    area = image.crop((x, y, x + cell_width, y + cell_height))
    pixels = np.array(area)
    average_color = pixels.mean(axis=(0, 1))
    return tuple(average_color.astype(int))

def classify_color(rgb):
    # Define the RGB codes for different gems
    # Gems may be special or normal
    color_thresholds = {
        'r': [(143, 27, 43), (191, 78, 84)],
        'b': [(35, 86, 133)],
        'y': [(118, 113, 52), (147, 116, 64)],
        'p': [(101, 55, 112)],
        'o': [(139, 87, 48), (186, 133, 82)],
        'g': [(35, 127, 59)],
        'w': [(125, 129, 132), (255,255,255)]
    }
    closest_color = None
    min_distance = float('inf')
    # Find the closest color match
    for color, rgb_values in color_thresholds.items():
        for threshold in rgb_values:
            distance = np.linalg.norm(np.array(rgb) - np.array(threshold))
            if distance < min_distance:
                min_distance = distance
                closest_color = color

    return closest_color

# Create a gem matrix
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

# Game Board region for splitscreen
board_region = (367, 245, 340, 350)

rows, cols = 8, 8

image = pyautogui.screenshot(region=board_region)

cell_width = image.width // cols
cell_height = image.height // rows

color_matrix = create_color_matrix(image, rows, cols, cell_width, cell_height)

# Checks left side solutions

def check_left_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    n = 0
    x = 0
    r = 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_left_2 = mat[i][j - 2] if j > 1 else None
            current_left_3 = mat[i][j - 3] if j > 2 else None
            current_left_up = mat[i - 1][j - 1] if (j > 0) and (i > 0) else None
            current_left_down = mat[i + 1][j - 1] if (j > 0) and (i < 7) else None
            current_left_up2 = mat[i - 2][j - 1] if (j > 0) and (i > 1) else None
            current_left_down2 = mat[i + 2][j - 1] if (j > 0) and (i < 6) else None
            if (current_left_2 == current) and (current_left_3 == current):
                return True, i, j, current
            elif (current_left_up == current) and (current_left_down == current):
                return True, i, j, current
            elif (current_left_up == current) and (current_left_up2 == current):
                return True, i, j, current
            elif (current_left_down == current) and (current_left_down2 == current):
                return True, i, j, current
            else:
                continue
    return False, n, x, r


# Checks right side solutions

def check_right_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    n = 0
    x = 0
    r = 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_right_2 = mat[i][j + 2] if j < cols - 2 else None
            current_right_3 = mat[i][j + 3] if j < cols - 3 else None
            current_right_up = mat[i - 1][j + 1] if (j < 7) and (i > 0) else None
            current_right_down = mat[i + 1][j + 1] if (j < 7) and (i < 7) else None
            current_right_up2 = mat[i - 2][j + 1] if (j < 7) and (i > 1) else None
            current_right_down2 = mat[i + 2][j + 1] if (j < 7) and (i < 6) else None
            if (current_right_2 == current) and (current_right_3 == current):
                return True, i, j, current
            elif (current_right_up == current) and (current_right_down == current):
                return True, i, j, current
            elif (current_right_up == current) and (current_right_up2 == current):
                return True, i, j, current
            elif (current_right_down == current) and (current_right_down2 == current):
                return True, i, j, current
            else:
                continue
    return False, n, x, r


# Checks top side solutions

def check_up_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    n = 0
    x = 0
    r = 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_up_2 = mat[i - 2][j] if i > 1 else None
            current_up_3 = mat[i - 3][j] if i > 2 else None
            current_left_up = mat[i - 1][j - 1] if (j > 0) and (i > 0) else None
            current_right_up = mat[i - 1][j + 1] if (j < 7) and (i > 0) else None
            current_left2_up = mat[i - 1][j - 2] if (j > 1) and (i > 0) else None
            current_right2_up = mat[i - 1][j + 2] if (j < 6) and (i > 0) else None
            if (current_up_2 == current) and (current_up_3 == current):
                return True, i, j, current
            elif (current_right_up == current) and (current_right2_up == current):
                return True, i, j, current
            elif (current_left_up == current) and (current_left2_up == current):
                return True, i, j, current
            elif (current_left_up == current) and (current_right_up == current):
                return True, i, j, current
            else:
                continue
    return False, n, x, r


# Checks bottom side solutions

def check_down_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    n = 0
    x = 0
    r = 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_down_2 = mat[i + 2][j] if i < rows - 2 else None
            current_down_3 = mat[i + 3][j] if i < rows - 3 else None
            current_left_down = mat[i + 1][j - 1] if (j > 0) and (i < 7) else None
            current_right_down = mat[i + 1][j + 1] if (j < 7) and (i < 7) else None
            current_left2_down = mat[i + 1][j - 2] if (j > 1) and (i < 7) else None
            current_right2_down = mat[i + 1][j + 2] if (j < 6) and (i < 7) else None
            if (current_down_2 == current) and (current_down_3 == current):
                return True, i, j, current
            elif (current_right_down == current) and (current_right2_down == current):
                return True, i, j, current
            elif (current_left_down == current) and (current_left2_down == current):
                return True, i, j, current
            elif (current_left_down == current) and (current_right_down == current):
                return True, i, j, current
            else:
                continue
    return False, n, x, r


# Moves the cursor to solve gems

def solve_puzzle(matrix):
    bool_left, i, j, current = check_left_solutions(matrix)
    bool_right, i1, j1, current1 = check_right_solutions(matrix)
    bool_up, i2, j2, current2 = check_up_solutions(matrix)
    bool_down, i3, j3, current3 = check_down_solutions(matrix)
    if bool_left == True:
        print(f'swap {current} left {i},{j}')
        pyautogui.moveTo(390 + (j*40), 270 + (i*40))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((390 + (j*40))-40, (270 + (i*40)), 0.5)
    elif bool_right == True:
        print(f'swap {current1} right {i1},{j1}')
        pyautogui.moveTo(390 + (j1 * 40), 270 + (i1 * 40))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((390 + (j1 * 40)) + 40, (270 + (i1 * 40)), 0.5)
    elif bool_up == True:
        print(f'swap {current2} up {i2},{j2}')
        pyautogui.moveTo(390 + (j2 * 40), 270 + (i2 * 40))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((390 + (j2 * 40)), (270 + (i2 * 40)) - 40, 0.5)
    elif bool_down == True:
        print(f'swap {current3} down {i3},{j3}')
        pyautogui.moveTo(390 + (j3 * 40), 270 + (i3 * 40))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((390 + (j3 * 40)), (270 + (i3 * 40)) + 40, 0.5)

# Calls helpers and plays Bejeweled
def play():
    try: 
        while True:
            time.sleep(3)
            board_region = (367, 245, 340, 350)
    
            rows, cols = 8, 8  # Adjust based on the game grid
    
            image = pyautogui.screenshot(region=board_region)
    
            cell_width = image.width // cols
            cell_height = image.height // rows
    
            color_matrix = create_color_matrix(image, rows, cols, cell_width, cell_height)
    
            for row in color_matrix:
                formatted_row = ', '.join(row)  # Join elements with a comma and space
                print(f'[{formatted_row}]')
    
            solve_puzzle(color_matrix)
    except KeyboardInterrupt:
        print('Program Terminated.')
        return

play()
