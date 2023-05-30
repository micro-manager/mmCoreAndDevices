import sys
import numpy as np

sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\micro-manager\\mmCoreAndDevices\\DeviceAdapters\\MM_pydevice')
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\hardware\\generic_binding')
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\wavefront_shaping_python')
sys.path.append('C:\\Users\\Jeroen Doornbos\\Documents\\wfs_current\\wavefront_shaping_python\\simulation')
from SSA import SSA
from Fourier import FourierDualRef
from SLMwrapper import SLM, set_circular_geometry
from base_device_properties import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options
#from WFS_functions import manual_slm_setup, select_roi, take_image, single_capt, point_capt, make_point_mean, select_point, example_mean, full_experiment_ssa
from WFS_functions import manual_slm_setup,select_point,full_experiment_fourier, full_experiment_ssa, wfs_procedure,slm_setup
from Simulation import SimulatedWFS
from galvo_scanner import Camera

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

        return np.reshape(self.camera_object.image, [self.camera_object._height,self.camera_object._width],order='C')

    def on_execute(self,value):
        if value:

            self.optimised_wf, _ = wfs_procedure(self.algorithm, self.slm_object,self.take_image)

        return value

    def on_optimised_wf(self,value):
        if value:
            plt.imshow(self.optimised_wf)
            self.slm_object.set_data(self.optimised_wf)
            self.slm_object.update(10)
            plt.show()
        return value

    def on_flat_wf(self,value):
        if value:
            self.slm_object.set_data(0)
            self.slm_object.update(10)

        return value



    active_slm = False
    slm = None


    algorithm = object_property()
    optimised_wf = None
    slm_object = object_property()
    camera_object = object_property()
    execute = bool_property(default=0, on_update=on_execute)
    show_optimised_wavefront = bool_property(default=0, on_update=on_optimised_wf)
    show_flat_wavefront = bool_property(default=0, on_update=on_flat_wf)
