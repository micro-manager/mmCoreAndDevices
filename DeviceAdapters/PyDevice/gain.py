import nidaqmx as ni
from nidaqmx.constants import LineGrouping
import numpy as np
from base_device_properties import float_property, int_property, string_property, object_property, base_property, bool_property, parse_options
import time
class Gain():

    def __init__(self,**kwargs):
        parse_options(self,kwargs)

    def set_gain(self,value):
        with ni.Task() as write_task:
            aochan = write_task.ao_channels.add_ao_voltage_chan(self.port_ao)
            aochan.ao_min = 0 
            aochan.ao_max = 0.9

            write_task.write(value)

        return value

    def check_overload(self):
        with ni.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.port_ai)
            in_stream = task.in_stream


            data = in_stream.read(number_of_samples_per_channel=1)
            if data > 2.5:
                self.overload = True
            else:
                self.overload = False

            return self.overload

        return overload

    def reset(self,value):
        if value:
            with ni.Task() as task:
                task.do_channels.add_do_chan(self.port_do,line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
                task.write([True])
                time.sleep(1)
                task.write([False])
        return value


    port_ao = string_property(default="Dev4/ao0")
    port_ai = string_property(default="Dev4/ai0")
    port_do = string_property(default="Dev4/port0/line0")

    reset = bool_property(default = 0, on_update = reset)
    gain = float_property(min=0, max = 0.9, default = 0,on_update = set_gain)

gain = Gain()
gain.reset = 1
gain.gain = 0.7
gain.check_overload()