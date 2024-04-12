import pymmcore
import os

mm_dir = "C:/Program Files/Micro-Manager-2.0/"
os.environ["PATH"] += os.pathsep.join(["", mm_dir])  # advisable on Windows


def test_generic_device():
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([mm_dir])
    mmc.loadSystemConfiguration("device.cfg")
    assert mmc.getProperty("some_device", "Integer") == '4'


def test_camera():
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([mm_dir])
    mmc.loadSystemConfiguration("camera.cfg")
    mmc.setProperty("cam", "Width", 121)
    mmc.setProperty("cam", "Height", 333)
    mmc.snapImage()
    frame = mmc.getImage()
    assert frame.shape == (333, 121)


def test_microscope():
    mmc = pymmcore.CMMCore()
    mmc.setDeviceAdapterSearchPaths([mm_dir])
    mmc.loadSystemConfiguration("microscope.cfg")
    mmc.snapImage()
    frame = mmc.getImage()
