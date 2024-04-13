""" Sample microscope
=======================
This script simulates a microscopic imaging system, generating a random noise image as a mock source and capturing it
through a microscope with adjustable magnification, numerical aperture, and wavelength. It visualizes the original and
processed images dynamically, demonstrating how changes in optical parameters affect image quality and resolution.
"""
import astropy.units as u
import numpy as np
from openwfs.plot_utilities import grab_and_show, imshow
from openwfs.simulation import Microscope, StaticSource
from openwfs.utilities import set_pixel_size

# Parameters that can be altered
img_size_x = 1024
# Determines how wide the image is.

img_size_y = 1024
# Determines how high the image is.

magnification = 40
# magnification from object plane to camera.

numerical_aperture = 0.85
# numerical aperture of the microscope objective

wavelength = 532.8 * u.nm
# wavelength of the light, different wavelengths are possible, units can be adjusted accordingly.

pixel_size = 6.45 * u.um
# Size of the pixels on the camera

camera_resolution = (256, 256)
# number of pixels on the camera

p_limit = 100
# Number of iterations. Influences how quick the 'animation' is complete.

# Code
img = set_pixel_size(
    np.maximum(np.random.randint(-10000, 100, (img_size_y, img_size_x), dtype=np.int16), 0),
    60 * u.nm)
src = StaticSource(img)
mic = Microscope(src, magnification=magnification, numerical_aperture=numerical_aperture, wavelength=wavelength)

# simulate shot noise in an 8-bit camera with auto-exposure:
cam = mic.get_camera(shot_noise=True, digital_max=255, data_shape=camera_resolution, pixel_size=pixel_size)
stage = mic.xy_stage

# todo: remove when fixed in openwfs
cam.__class__.exposure = cam.__class__.duration
stage.__class__.step_size_x_um = float
stage.step_size_x_um = 0.001
stage.__class__.step_size_y_um = float
stage.step_size_y_um = 0.001
devices = {'camera': cam, 'stage': stage}

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bootstrap import PyDevice

    c_device = PyDevice(devices['camera'])
    print(c_device)
    assert c_device.device_type == 'Camera'
    s_device = PyDevice(devices['stage'])
    print(s_device)
    assert s_device.device_type == 'XYStage'
