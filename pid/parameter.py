import numpy as np
import math


def hexStr(response):
    if response is not None:
        hex_string = ' '.join(f'{byte:02x}' for byte in response)
        return hex_string
    return None
        

class Parameters():
    pi = math.pi
    PI = pi

    uintDegreeEncoder = 32768/360
    rotateEncoder = 32768

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

    RTS_PIN = 11

    anglesPerPixel_X = horizontal_FOV / HD_Width

    