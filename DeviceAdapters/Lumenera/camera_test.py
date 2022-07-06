"""
This script tests a camera by running through various API functions and properties

It can be runs using pycromanager, which needs to be set up by following the instructions here:
https://pycro-manager.readthedocs.io/en/latest/setup.html

Additionally, the end part requires matplotlib to be installed for plotting images
"""

from pycromanager import Core
import numpy as np
import time

### Utility functions
def snap_and_get():
    core.snap_image()
    tagged_image = core.get_tagged_image()
    if core.get_number_of_components() == 4:
        # rgba
        new_shape = [tagged_image.tags['Height'],
                                  tagged_image.tags['Width'], 4]
    elif core.get_number_of_components() == 3:
        # rgb
        new_shape = [tagged_image.tags['Height'],
                                  tagged_image.tags['Width'], 3]
    else:
        # monochrome
        new_shape = [tagged_image.tags['Height'],
                                  tagged_image.tags['Width']]
    pixels = np.reshape(tagged_image.pix,
                        newshape=new_shape)
    return pixels

to_python_list = lambda str_vec:  [str_vec.get(i) for i in range(str_vec.size())]

def get_property_limits(prop_name):
    return (core.get_property_lower_limit(camera_name, prop_name),
        core.get_property_upper_limit(camera_name, prop_name))

def test_image(prop_name, prop_value):
    core.set_property(camera_name, prop_name, prop_value)
    if core.get_property(camera_name, prop_name) != prop_value:
        print("\tRead {} after writing {}".format(
            prop_value, core.get_property(camera_name, prop_name))
        )
    image = snap_and_get()
    if len(image.shape) == 2:
        print("\tmean pixel value with {} {}".format(prop_value, prop_name),
          np.mean(image))
    else:
        print("\tmean RGB pixel value with {} {}".format(prop_value, prop_name),
              "{:.2f}, {:.2f}, {:.2f}".format(*
                            np.mean(image[..., :3], axis=(0,1))))

def test_property_limits(prop_name):
    limits = get_property_limits(prop_name)
    initial = core.get_property(camera_name, prop_name)
    t = core.get_property_type(camera_name, "Contrast")
    if t.to_string() == "Float":
        prop_type = float
    elif t.to_string() == "Integer":
        prop_type = int
    else:
        prop_type = str
    limits = (prop_type(limits[0]), prop_type(limits[1]))
    ############### Compensate for set/get issue in LuXApps
    if prop_type == float:
        delta = (limits[1] - limits[0]) *0.1
        limits = (limits[0] + delta, limits[1] - delta)
    ###############
    test_image(prop_name, limits[0])
    test_image(prop_name, limits[1])
    core.set_property(camera_name, prop_name, initial)

### Setup
# Connect to micro-manager core
core = Core()
camera_name = core.get_camera_device()

# Get and print names of all properties
props = to_python_list(core.get_device_property_names(camera_name))
print("Available properties: ", ",".join(props), "\n\n")


### Test camera API
# Test effect of exposure on image brightness
print("Testing exposure API:")
for exposure in [1000, 100, 10]:
    core.set_exposure(exposure)
    image = snap_and_get()
    print("\tmean pixel value with {} ms exposure".format(exposure),
          np.mean(image))

# Test setting of ROI
try:
    # full size
    core.clear_roi()
    core.set_roi(camera_name, 0, 0, core.get_image_width(), core.get_image_height())
    print("Full sensor ROI: ", core.get_image_width(), core.get_image_height())
    # half size
    core.set_roi(camera_name, 0, 0, core.get_image_width() // 2, core.get_image_height() // 2)
    print("Half sensor ROI: ", core.get_image_width(), core.get_image_height())
    core.clear_roi()
except:
    raise Exception("Couldn't set ROI to full sensor")

# Test speed of sequence acquisition
core.set_exposure(100)
start_time = time.time()
num_images = 10
core.start_sequence_acquisition(num_images, 0, True)
for i in range(10):
    while True:
        try:
            core.pop_next_image()
            break
        except:
            continue
core.stop_sequence_acquisition()
end_time = time.time()
print("Sequence acquisition speed test:\n"
      "\tacquired 10 images with 100 ms exposure in {} seconds".format(end_time - start_time),
      "{:.2f} ms per image".format((end_time - start_time) / num_images * 1000))



### Test camera properties
print("Testing camera properties")
core.set_property(camera_name, "Gain", 1.0)
core.set_exposure(10)

# Binning
print("Binning")
for bin_value in to_python_list(core.get_allowed_property_values(camera_name, "Binning")):
    core.set_property(camera_name, "Binning", bin_value)
    print("\tBinning value: {} \t\t Image size: {}".format(bin_value, snap_and_get().shape))

for image_property in [
    "Contrast", "Gain",
    "Gamma", "Hue", "RGB Blue Gain",
    "RGB Green1 Gain", "RGB Green2 Gain", "RGB Red Gain",
    "Saturation", "White Balance Target (Blue)",
    "White Balance Target (Green)", "White Balance Target (Red)"]:
    if core.has_property(camera_name, image_property):
        print(image_property)
        test_property_limits(image_property)


### Test properties that require plots
import matplotlib.pyplot as plt

def show_image_test(prop_name):
    vals = to_python_list(core.get_allowed_property_values(camera_name, prop_name))
    fig, ax = plt.subplots(len(vals), 1, figsize=(2, 2*len(vals)))
    initial_val = core.get_property(camera_name, prop_name)
    for i, val in enumerate(vals):
        core.set_property(camera_name, prop_name, val)
        image = snap_and_get()
        ax[i].set_title("{}: {}".format(prop_name, val))
        ax[i].imshow(image)
        ax[i].set_axis_off()
    core.set_property(camera_name, prop_name, initial_val)

for prop in [
        "Flipping",
        "Mirror"]:
    show_image_test(prop)
plt.show()


