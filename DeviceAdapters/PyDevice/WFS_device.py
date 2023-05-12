import sys
import numpy as np

sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\micro-manager\\mmCoreAndDevices\\DeviceAdapters\\MM_pydevice')
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\hardware\\generic_binding')
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\wavefront_shaping_python')

from SSA import SSA
from Fourier import FourierDualRef
from SLMwrapper import SLM, set_circular_geometry, test
from test_cam import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options
#from WFS_functions import manual_slm_setup, select_roi, take_image, single_capt, point_capt, make_point_mean, select_point, example_mean, full_experiment_ssa
from WFS_functions import manual_slm_setup,select_point,full_experiment_fourier, full_experiment_ssa, wfs_procedure
from galvo_scanner import Camera
class SLM_device:

    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True

    def make_slm(self):
        self.s = SLM(self._slm_number)
        return self.s
    def delete_slm(self):
        self.s.destroy()

    slm_number = int_property(min = 0,default=0)


class SSA_device:

    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True

    def init_wf(self):
        return np.zeros([self._n_slm_fields, self._n_slm_fields])

    def algorithm(self):
        return SSA(self._phase_steps,self.init_wf())
    # def set_devices(self,slm,get_feedback):
    #     self.slm = slm
    #     self.feedback = get_feedback


    phase_steps = int_property(min = 1, default=8)
    n_slm_fields = int_property(min=1, default=1)

class fourier_device:

    def __init__(self, **kwargs):
        parse_options(self, kwargs)
        self.resized = True

    def init_wf(self):
        return np.zeros([1056, 1056])
    def algorithm(self):
        kx = np.arange(self._kx_angles_min, self._kx_angles_max, self._kx_angles_stepsize)
        ky = np.arange(self._ky_angles_min, self._ky_angles_max, self._ky_angles_stepsize)
        return FourierDualRef(self._phase_steps, self.init_wf(), kx, ky, self._overlap_coeficient, 0)



    phase_steps = int_property(min = 1, default=8)
    kx_angles_min = int_property(default=-6)
    kx_angles_max = int_property(default=6)
    kx_angles_stepsize = int_property(default=6)
    ky_angles_min = int_property(default=-6)
    ky_angles_max = int_property(default=6)
    ky_angles_stepsize = int_property(default=6)
    overlap_coeficient = float_property(min=0, max=1, default=0.1)

import matplotlib.pyplot as plt
class WFS:

    def __init__(self, **kwargs):
        parse_options(self, kwargs)


    def _ValueError(self, msg):
        raise ValueError(msg)

    def take_image(self):
        self.camera_object.trigger()
        self.camera_object.wait()
        return self.camera_object.image

    def on_execute(self,value):



        if value:
            if self.slm_object == None:
                self._ValueError(f"A SLM object needs to be connected in order to wavefront shape")

            if self.camera_object == None:
                self._ValueError(f"A Camera object needs to be added in order to wavefront shape")

            if self.algorithm == None:
                self._ValueError(f"An algorithm object needs to be added in order to wavefront shape")

            wfs_procedure(self.algorithm, self.slm_object.make_slm(),self.take_image)


        self._execute = False
        return value

    algorithm = object_property()
    slm_object = object_property()
    camera_object = object_property()
    execute = bool_property(default=0, on_update=on_execute)