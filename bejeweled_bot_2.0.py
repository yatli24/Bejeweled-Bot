import pyautogui
from PIL import ImageGrab
import numpy as np
import time
from PIL import Image


# Average color helper
def get_average_color(image, x, y, cell_width, cell_height):
    area = image.crop((x, y, x + cell_width, y + cell_height))
    pixels = np.array(area)
    average_color = pixels.mean(axis=(0, 1))
    return tuple(average_color.astype(int))

# Classify a cell by color
def classify_color(rgb):
    # Manually define the RGB codes for different gems
    # Gems may be flaming, shining, special, or normal
    color_thresholds = {
        'r': [(143, 27, 43), (191, 78, 84), (123,12,35), (191, 125, 158)],
        'b': [(35, 86, 133), (92, 120, 129), (23,58, 110), (62, 91, 126), (121, 182, 208)],
        'y': [(118, 113, 52), (147, 116, 64), (91, 72, 27), (99, 76, 38), (126,95,57), (155, 166, 153)],
        'p': [(101, 55, 112), (79, 12, 86), (128,54,111)],
        'o': [(139, 87, 48), (186, 133, 82), (121, 64, 39), (152, 98, 58)],
        'g': [(35, 127, 59), (36,106, 56), (67, 137, 71)],
        'w': [(125, 129, 132), (101,91,106), (184, 174, 159)]
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

# Create a gem matrix for the greedy algorithm to solve
def create_gem_matrix(image, rows, cols, cell_width, cell_height):
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

# Checks left side solutions

def check_left_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_left_2 = mat[i][j - 2] if j > 1 else None
            current_left_3 = mat[i][j - 3] if j > 2 else None
            current_left_4 = mat[i][j-4] if j > 3 else None
            current_left_up = mat[i - 1][j - 1] if (j > 0) and (i > 0) else None
            current_left_down = mat[i + 1][j - 1] if (j > 0) and (i < 7) else None
            current_left_up2 = mat[i - 2][j - 1] if (j > 0) and (i > 1) else None
            current_left_down2 = mat[i + 2][j - 1] if (j > 0) and (i < 6) else None
            current_left_up3 = mat[i - 3][j - 1] if (j > 0) and (i > 2) else None
            current_left_down3 = mat[i + 3][j - 1] if (j > 0) and (i < 5) else None

            if (current_left_2 == current) and (current_left_3 == current) and (current_left_4 == current):
                return True, i, j, current, 1
            elif (current_left_up3 == current) and (current_left_up2 == current) and (current_left_up == current):
                return True, i, j, current, 1
            elif (current_left_down3 == current) and (current_left_down2 == current) and (current_left_down == current):
                return True, i, j, current, 1
            elif(current_left_up == current) and (current_left_up2 == current) and (current_left_down == current):
                return True, i, j, current, 1
            elif (current_left_up == current) and (current_left_down == current) and (current_left_down2 == current):
                return True, i, j, current, 1
            elif (current_left_2 == current) and (current_left_3 == current):
                return True, i, j, current, 0
            elif (current_left_up == current) and (current_left_down == current):
                return True, i, j, current, 0
            elif (current_left_up == current) and (current_left_up2 == current):
                return True, i, j, current, 0
            elif (current_left_down == current) and (current_left_down2 == current):
                return True, i, j, current, 0
            else:
                continue
    return False, 0, 0, 0, 0

# Checks right side solutions

def check_right_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_right_2 = mat[i][j + 2] if j < 6 else None
            current_right_3 = mat[i][j + 3] if j < 5 else None
            current_right_4 = mat[i][j + 4] if j < 4 else None
            current_right_up = mat[i - 1][j + 1] if (j < 7) and (i > 0) else None
            current_right_down = mat[i + 1][j + 1] if (j < 7) and (i < 7) else None
            current_right_up2 = mat[i - 2][j + 1] if (j < 7) and (i > 1) else None
            current_right_down2 = mat[i + 2][j + 1] if (j < 7) and (i < 6) else None
            current_right_up3 = mat[i - 3][j + 1] if (j < 7) and (i > 2) else None
            current_right_down3 = mat[i + 3][j + 1] if (j < 7) and (i < 5) else None

            if (current_right_2 == current) and (current_right_3 == current) and (current_right_4 == current):
                return True, i, j, current, 1
            elif (current_right_up3 == current) and (current_right_up2 == current) and (current_right_up == current):
                return True, i, j, current, 1
            elif (current_right_down3 == current) and (current_right_down2 == current) and (current_right_down == current):
                return True, i, j, current, 1
            elif (current_right_up == current) and (current_right_up2 == current) and (current_right_down == current):
                return True, i, j, current, 1
            elif (current_right_up == current) and (current_right_down == current) and (current_right_down2 == current):
                return True, i, j, current, 1
            if (current_right_2 == current) and (current_right_3 == current):
                return True, i, j, current, 0
            elif (current_right_up == current) and (current_right_down == current):
                return True, i, j, current, 0
            elif (current_right_up == current) and (current_right_up2 == current):
                return True, i, j, current, 0
            elif (current_right_down == current) and (current_right_down2 == current):
                return True, i, j, current, 0
            else:
                continue
    return False, 1, 0, 0, 0


# Checks top side solutions

def check_up_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_up_2 = mat[i - 2][j] if i > 1 else None
            current_up_3 = mat[i - 3][j] if i > 2 else None
            current_up_4 = mat[i - 4][j] if i > 3 else None
            current_left_up = mat[i - 1][j - 1] if (j > 0) and (i > 0) else None
            current_right_up = mat[i - 1][j + 1] if (j < 7) and (i > 0) else None
            current_left2_up = mat[i - 1][j - 2] if (j > 1) and (i > 0) else None
            current_right2_up = mat[i - 1][j + 2] if (j < 6) and (i > 0) else None
            current_left3_up = mat[i - 1][j - 3] if (j > 2) and (i > 0) else None
            current_right3_up = mat[i - 1][j + 3] if (j < 5) and (i > 0) else None

            if (current_up_4 == current) and (current_up_3 == current) and (current_up_2 == current):
                return True, i, j, current, 1
            elif (current_left3_up == current) and (current_left2_up == current) and (current_left_up == current):
                return True, i, j, current, 1
            elif (current_right3_up == current) and (current_right2_up == current) and (current_right_up == current):
                return True, i, j, current, 1
            elif (current_right_up == current) and (current_right2_up == current) and (current_left_up == current):
                return True, i, j, current, 1
            elif (current_right_up == current) and (current_left_up == current) and (current_left2_up == current):
                return True, i, j, current, 1
            elif (current_up_2 == current) and (current_up_3 == current):
                return True, i, j, current, 0
            elif (current_right_up == current) and (current_right2_up == current):
                return True, i, j, current, 0
            elif (current_left_up == current) and (current_left2_up == current):
                return True, i, j, current, 0
            elif (current_left_up == current) and (current_right_up == current):
                return True, i, j, current, 0
            else:
                continue
    return False, 0, 1, 0, 0


# Checks bottom side solutions

def check_down_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_down_2 = mat[i + 2][j] if i < 6 else None
            current_down_3 = mat[i + 3][j] if i < 5 else None
            current_down_4 = mat[i+4][j] if i < 4 else None
            current_left_down = mat[i + 1][j - 1] if (j > 0) and (i < 7) else None
            current_right_down = mat[i + 1][j + 1] if (j < 7) and (i < 7) else None
            current_left2_down = mat[i + 1][j - 2] if (j > 1) and (i < 7) else None
            current_right2_down = mat[i + 1][j + 2] if (j < 6) and (i < 7) else None
            current_left3_down = mat[i + 1][j - 3] if (j > 2) and (i < 7) else None
            current_right3_down = mat[i + 1][j + 3] if (j < 5) and (i < 7) else None

            if (current_down_4 == current) and (current_down_3 == current) and (current_down_2 == current):
                return True, i, j, current, 1
            elif (current_left3_down == current) and (current_left2_down == current) and (current_left_down == current):
                return True, i, j, current, 1
            elif (current_right3_down == current) and (current_right2_down == current) and (current_right_down == current):
                return True, i, j, current, 1
            elif (current_right_down == current) and (current_right2_down == current) and (current_left_down == current):
                return True, i, j, current, 1
            elif (current_right_down == current) and (current_left_down == current) and (current_left2_down == current):
                return True, i, j, current, 1
            elif (current_down_2 == current) and (current_down_3 == current):
                return True, i, j, current, 0
            elif (current_right_down == current) and (current_right2_down == current):
                return True, i, j, current, 0
            elif (current_left_down == current) and (current_left2_down == current):
                return True, i, j, current, 0
            elif (current_left_down == current) and (current_right_down == current):
                return True, i, j, current, 0
            else:
                continue
    return False, 0, 0, 1, 0


# Moves the cursor to solve gems

def solve_puzzle(matrix, cell_width, board_left_x, board_left_y):
    bool_left, i, j, current, four = check_left_solutions(matrix)
    bool_right, i1, j1, current1, four1 = check_right_solutions(matrix)
    bool_up, i2, j2, current2, four2 = check_up_solutions(matrix)
    bool_down, i3, j3, current3, four3 = check_down_solutions(matrix)
    cursor_x = board_left_x + (cell_width // 2)
    cursor_y = board_left_y + (cell_width // 2)
    if four == 1:
        print(f'swap {current} left {i},{j}')
        pyautogui.moveTo(cursor_x + (j * cell_width), cursor_y + (i * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j * cell_width)) - cell_width, (cursor_y + (i * cell_width)), 0.5)
    elif four1 == 1:
        print(f'swap {current1} left {i1},{j1}')
        pyautogui.moveTo(cursor_x + (j1 * cell_width), cursor_y + (i1 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j1 * cell_width)) + cell_width, (cursor_y + (i1 * cell_width)), 0.5)
    elif four2 == 1:
        print(f'swap {current2} left {i2},{j2}')
        pyautogui.moveTo(cursor_x + (j2 * cell_width), cursor_y + (i2 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j2 * cell_width)), (cursor_y + (i2 * cell_width)) - cell_width, 0.5)
    elif four3 == 1:
        print(f'swap {current3} left {i3},{j3}')
        pyautogui.moveTo(cursor_x + (j3 * cell_width), cursor_y + (i3 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j3 * cell_width)), (cursor_y + (i3 * cell_width)) + cell_width, 0.5)
    elif bool_left == True:
        print(f'swap {current} left {i},{j}')
        pyautogui.moveTo(cursor_x + (j * cell_width), cursor_y + (i * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j * cell_width)) - cell_width, (cursor_y + (i * cell_width)), 0.5)
    elif bool_right == True:
        print(f'swap {current1} right {i1},{j1}')
        pyautogui.moveTo(cursor_x + (j1 * cell_width), cursor_y + (i1 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j1 * cell_width)) + cell_width, (cursor_y + (i1 * cell_width)), 0.5)
    elif bool_up == True:
        print(f'swap {current2} up {i2},{j2}')
        pyautogui.moveTo(cursor_x + (j2 * cell_width), cursor_y + (i2 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j2 * cell_width)), (cursor_y + (i2 * cell_width)) - cell_width, 0.5)
    elif bool_down == True:
        print(f'swap {current3} down {i3},{j3}')
        pyautogui.moveTo(cursor_x + (j3 * cell_width), cursor_y + (i3 * cell_width))
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo((cursor_x + (j3 * cell_width)), (cursor_y + (i3 * cell_width)) + cell_width, 0.5)

# Function to call helpers and play Bejeweled
def play():
    try:
        while True:
            # Give the board time to reset
            time.sleep(1.5)
            # Set game region locally (Hard Coded)
            board_top_left_x = 367
            board_top_left_y = 245
            board_width = 340
            board_height = 350
            board_region = (board_top_left_x, board_top_left_y, board_width, board_height)

            image = pyautogui.screenshot(region=board_region)

            rows, cols = 8, 8
            cell_width = image.width // cols
            cell_height = image.height // rows

            gem_matrix = create_gem_matrix(image, rows, cols, cell_width, cell_height)

            # Print gem matrix
            for row in gem_matrix:
                formatted_row = ', '.join(row)
                print(f'[{formatted_row}]')

            solve_puzzle(color_matrix, cell_width, board_top_left_x, board_top_left_y)

    except KeyboardInterrupt:
        print('Program Terminated.')
        return

# Run the bot
if __name__== "__main__":
    play()
