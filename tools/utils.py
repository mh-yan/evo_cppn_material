import numpy as np
import os
import matplotlib.pyplot as plt
import tools.shape as shape


def d1(x, y):
    return x - 0.5


def d2(x, y):
    return (y - (1.0 / 3**0.5) * x) / ((1 + 1.0 / 3) ** 0.5)


def d3(x, y):
    return (y + (1 / 3**0.5) * x - 1 / 3**0.5) / ((1 + 1 / 3) ** 0.5)


def d4(x, y):
    return (y + (3**0.5) * x - 3**0.5) / 2


def diag_line(x, y):
    return (y + (3**0.5) * x - 3**0.5) / 2


def vf1(x):
    return -(3**0.5) / 3 * x + (3**0.5) / 3


def vf2(x):
    return ((3**0.5) / 3) * x


def dig(x):
    return -(3**0.5) * x + 3**0.5


def normalize(x, b=None, u=None):
    if b != None:
        x -= b
    else:
        x -= np.min(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        if u != None:
            x /= u
        else:
            x /= np.max(x)
    x = np.nan_to_num(x)
    # adjust the range to [x1,x2]
    x *= 2
    x -= 1
    return x


def scale(x, min=0, max=1):
    # scale x to [min, max]
    x = np.nan_to_num(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    return x


def rotate_point_around_pivot(px, py, cx, cy, theta):
    # Convert angle from degrees to radians
    theta = np.radians(theta)

    # Step 1: Translate the point to the origin
    translated_x = px - cx
    translated_y = py - cy

    # Step 2: Apply the rotation matrix
    rotated_x = translated_x * np.cos(theta) - translated_y * np.sin(theta)
    rotated_y = translated_x * np.sin(theta) + translated_y * np.cos(theta)

    # Step 3: Translate the point back
    new_x = rotated_x + cx
    new_y = rotated_y + cy

    return new_x, new_y


def reflect_point_about_line(px, py, a, b, c):
    # Step 1: Translate to make the line pass through the origin
    d = a * px + b * py + c
    translated_x = px
    translated_y = py

    # Step 2: Rotate the line to align with the x-axis
    theta = np.arctan2(b, a)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated_x = cos_theta * translated_x + sin_theta * translated_y
    rotated_y = -sin_theta * translated_x + cos_theta * translated_y

    # Step 3: Reflect about the x-axis
    reflected_x = rotated_x
    reflected_y = -rotated_y

    # Step 4: Rotate back
    final_x = cos_theta * reflected_x - sin_theta * reflected_y
    final_y = sin_theta * reflected_x + cos_theta * reflected_y

    return final_x, final_y


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def plotall(
    config,
    g_path,
    thresh,
    pcd,
    Tri,
    shapex,
    shapey,
    fixed_list,
    is_mesh,
    go_path,
    sympcd=None,
):

    g = shape.getshape(
        config,
        g_path,
        thresh,
        pcd,
        Tri,
        shapex,
        shapey,
        fixed_list,
        is_mesh,
        go_path,
        sympcd=sympcd,
    )


def clear_folder(folder):
    if not os.path.exists(folder):
        print(f"The folder '{folder}' does not exist.")
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clear_folder(file_path)  # 递归删除子文件夹
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
