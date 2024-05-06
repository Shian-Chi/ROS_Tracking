import numpy as np
import math


def calcVertical_FOV(diagonal_size=4.60, horizontal_fov_deg=77):
    # Convert horizontal FOV to radians
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    # Calculate horizontal and vertical sizes of the camera sensor
    horizontal_size = diagonal_size * math.cos(horizontal_fov_rad / 2)
    vertical_size = horizontal_size * (9 / 16)  # Assuming a 16:9 aspect ratio
    # Calculate half of the vertical FOV in radians
    vertical_fov_rad_half = math.atan(vertical_size / horizontal_size)
    # Convert half of the vertical FOV to degrees
    vertical_fov_deg = math.degrees(2 * vertical_fov_rad_half)
    return vertical_fov_deg
            

class Parameters():
    pi = math.pi
    PI = pi

    uintDegreeEncoder = np.array([32767/360], dtype="float")
    rotateEncoder = np.array([32768], dtype="uint16")

    MOTOR_FORWARD = 0
    MOTOR_REVERSE = 1

    # HD 720P
    HD_Width = 1280
    HD_Height = 720

    # FHD 1080P
    FHD_Width = 1920
    FHD_Height = 1080

    # QHD 2K
    QHD_Width = 2560
    QHD_Height = 1440

    # UHD 4K
    UHD_Width = 3280
    UHD_Height = 2464

    CMOS_SIZE = 4.60  # mm
    Focal_Length = 2.96  # mm
    horizontal_FOV = 77.0
    vertical_FOV = calcVertical_FOV(CMOS_SIZE, horizontal_FOV)

    RTS_PIN = 11

    anglesPerPixel_X = horizontal_FOV / HD_Width
    anglesPerPixel_Y = calcVertical_FOV(CMOS_SIZE, horizontal_FOV) / HD_Height