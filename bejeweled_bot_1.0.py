import pyautogui
import numpy as np
import time
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# TRAINING #

# Helper for image loading from folder
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# Average color helper
def get_average_color(image):
    average_color = cv2.mean(image)[:3]
    return average_color


# Helper for splitting gems from board
def split_screenshot(screenshot, grid_size):
    height, width, _ = screenshot.shape
    gem_height = height // grid_size[0]
    gem_width = width // grid_size[1]
    gems = []
    for row in range(grid_size[0]):
        row_gems = []
        for col in range(grid_size[1]):
            x_start = col * gem_width
            y_start = row * gem_height
            gem = screenshot[y_start:y_start + gem_height, x_start:x_start + gem_width]
            row_gems.append(gem)
        gems.append(row_gems)
    return gems


# Helper for gem classification
def classify_gem(image, model, labels):
    avg_color = get_average_color(image)
    avg_color = np.array(avg_color).reshape(1, -1)
    prediction = model.predict(avg_color)
    gem_color = labels[prediction[0]]
    return gem_color


# Create a color matrix for the greedy algorithm to solve
def create_gem_color_matrix(gems, model, labels):
    gem_color_matrix = []
    for row in gems:
        row_colors = []
        for gem in row:
            gem_color = classify_gem(gem, model, labels)
            row_colors.append(gem_color)
        gem_color_matrix.append(row_colors)
    return gem_color_matrix


# Define gem image folders
yellow_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\yellow'
red_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\red'
blue_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\blue'
orange_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\orange'
purple_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\purple'
white_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\white'
green_gem_folder = 'C:\\Users\\user\\dir\\BejeweledBot\\gem_images\\green'

# Load images separately
yellow_images = load_images(yellow_gem_folder)
red_images = load_images(red_gem_folder)
blue_images = load_images(blue_gem_folder)
orange_images = load_images(orange_gem_folder)
purple_images = load_images(purple_gem_folder)
white_images = load_images(white_gem_folder)
green_images = load_images(green_gem_folder)

# Get average color of each image for each color
yellow_avg_colors = [get_average_color(img) for img in yellow_images]
red_avg_colors = [get_average_color(img) for img in red_images]
blue_avg_colors = [get_average_color(img) for img in blue_images]
orange_avg_colors = [get_average_color(img) for img in orange_images]
purple_avg_colors = [get_average_color(img) for img in purple_images]
white_avg_colors = [get_average_color(img) for img in white_images]
green_avg_colors = [get_average_color(img) for img in green_images]

# Generate labels for images
yellow_labels = [0] * len(yellow_avg_colors)
red_labels = [1] * len(red_avg_colors)
blue_labels = [2] * len(blue_avg_colors)
orange_labels = [3] * len(orange_avg_colors)
purple_labels = [4] * len(purple_avg_colors)
white_labels = [5] * len(white_avg_colors)
green_labels = [6] * len(green_avg_colors)

# Define gem labels for each color
gem_labels = {
    0: 'y',
    1: 'r',
    2: 'b',
    3: 'o',
    4: 'p',
    5: 'w',
    6: 'g',
}

# Assign data and labels
data = yellow_avg_colors + red_avg_colors + blue_avg_colors + orange_avg_colors + purple_avg_colors + white_avg_colors + green_avg_colors
labels = yellow_labels + red_labels + blue_labels + orange_labels + purple_labels + white_labels + green_labels

data = np.array(data)
labels = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the k-nNeighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Capture the game board (Hardcoded Region)
def capture():
    screenshot = pyautogui.screenshot(region=(309, 406, 320, 320))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

# ALGORITHM #

# Check left side solutions
def check_left_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_left_2 = mat[i][j - 2] if j > 1 else None
            current_left_3 = mat[i][j - 3] if j > 2 else None
            current_left_4 = mat[i][j - 4] if j > 3 else None
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
            elif (current_left_up == current) and (current_left_up2 == current) and (current_left_down == current):
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


# Check right side solutions
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
            elif (current_right_down3 == current) and (current_right_down2 == current) and (
                    current_right_down == current):
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


# Check top side solutions
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


# Check bottom side solutions
def check_down_solutions(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            current = mat[i][j]
            current_down_2 = mat[i + 2][j] if i < 6 else None
            current_down_3 = mat[i + 3][j] if i < 5 else None
            current_down_4 = mat[i + 4][j] if i < 4 else None
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
            elif (current_right3_down == current) and (current_right2_down == current) and (
                    current_right_down == current):
                return True, i, j, current, 1
            elif (current_right_down == current) and (current_right2_down == current) and (
                    current_left_down == current):
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


# Moves the mouse cursor to solve gems using a greedy algorithm
def solve_puzzle(matrix, cell_width, board_left_x, board_left_y):
    bool_left, i, j, current, four = check_left_solutions(matrix)
    bool_right, i1, j1, current1, four1 = check_right_solutions(matrix)
    bool_up, i2, j2, current2, four2 = check_up_solutions(matrix)
    bool_down, i3, j3, current3, four3 = check_down_solutions(matrix)
    cursor_x = board_left_x + (cell_width // 2)
    cursor_y = board_left_y + (cell_width // 2)

    if four == 1 or four1 == 1 or four2 == 1 or four3 == 1:
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

    elif bool_up == True or bool_left == True or bool_right == True or bool_down == True:
        if bool_left == True:
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


# Function to run an indefinite loop until a keyboard interrupt
def play():
    try:
        while True:
            # Give the board time to reset
            time.sleep(1.5)
            
            # Define hardcoded region locally
            board_top_left_x = 309
            board_top_left_y = 406
            board_width = 320
            board_height = 320
            board_region = (board_top_left_x, board_top_left_y, board_width, board_height)

            rows, cols = 8, 8
            board = pyautogui.screenshot(region=board_region)

            cell_width = board.width // cols

            screenshot = capture()
            grid_size = (8, 8)
            gems = split_screenshot(screenshot, grid_size)
            gem_matrix = create_gem_color_matrix(gems, knn, gem_labels)

            for row in gem_matrix:
                formatted_row = ', '.join(row)
                print(f'[{formatted_row}]')

            solve_puzzle(gem_matrix, cell_width, board_top_left_x, board_top_left_y)

    except KeyboardInterrupt:
        print('Thank you for using Bejeweled Bot 1.0!')
        return
    print('Thank you for using Bejeweled Bot 1.0!')

# Run the bot
play()
