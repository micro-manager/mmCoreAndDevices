import nidaqmx as ni
from nidaqmx.constants import LineGrouping
from typing import Annotated
import time


class Gain:
    """
    A device that controls the voltage of a PMT gain using a NI data acquisition card.

    It is controlling a hamamatsu M9012 power supply, supplying power for a PMT H7422-40.

    It uses an analogue out connection to set the voltage,
    an analogue in channel to check the gain's self-protecting overload status,
    and a digital out that sends the signal to the gain to reset itself.

    It contains a boolean called reset that triggers the method on_reset.
    Ideally, we would like some button to execute methods in our devices,
    but this approach allows the user to execute methods in the device property manager.
    """
    def __init__(self, port_ao="Dev4/ao0", port_ai="Dev4/ai0", port_do="Dev4/port0/line0", reset=False, gain=0):
        self.port_ao = port_ao
        self.port_ai = port_ai
        self.port_do = port_do
        self._reset = reset
        self._gain = gain

    def set_gain(self, value):
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
                overload = True
            else:
                overload = False

            return overload

    def on_reset(self, value):
        if value:
            with ni.Task() as task:
                task.do_channels.add_do_chan(self.port_do, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
                task.write([True])
                time.sleep(1)
                task.write([False])

    @property
    def reset(self) -> bool:
        return self._reset

    @reset.setter
    def reset(self, value: bool):
        self.on_reset(value)
        self._reset = value

    @property
    def gain(self) -> Annotated[float, {'min': 0, 'max': 0.9}]:
        # The range of values is the hardware supplier's defined voltage range. Setting the range here for safety
        return self._gain

    @gain.setter
    def gain(self, value: float):
        self.set_gain(value)
        self._gain = value


devices = {'gain': Gain()}
