import pymmcore
import os

mm_dir = "C:/Program Files/Micro-Manager-2.0/"
os.environ["PATH"] += os.pathsep.join(["", mm_dir])  # advisable on Windows
config_dir = os.path.dirname(os.path.realpath(__file__))


def test_generic_device():
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([mm_dir])
    mmc.loadSystemConfiguration(os.path.join(config_dir, "generic_device.cfg"))
    assert mmc.getProperty("some_device", "Integer") == '4'


def test_camera():
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([mm_dir])
    mmc.loadSystemConfiguration(os.path.join(config_dir, "camera.cfg"))
    mmc.setProperty("Camera:cam", "Width", 121)
    mmc.setProperty("Camera:cam", "Height", 333)
    mmc.snapImage()
    frame = mmc.getImage()
    assert frame.shape == (333, 121)
