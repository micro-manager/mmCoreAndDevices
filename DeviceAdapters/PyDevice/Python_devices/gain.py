import nidaqmx as ni
from nidaqmx.constants import LineGrouping
from typing import Annotated
import time


class Gain:

    def __init__(self, port_ao="Dev4/ao0", port_ai="Dev4/ai0", port_do="Dev4/port0/line0", reset=False, gain=0):
        self._port_ao = port_ao
        self._port_ai = port_ai
        self._port_do = port_do
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
    def port_ao(self) -> str:
        return self._port_ao

    @port_ao.setter
    def port_ao(self, value: str):
        self._port_ao = value

    @property
    def port_ai(self) -> str:
        return self._port_ai

    @port_ai.setter
    def port_ai(self, value: str):
        self._port_ai = value

    @property
    def port_do(self) -> str:
        return self._port_do

    @port_do.setter
    def port_do(self, value: str):
        self._port_do = value

    @property
    def reset(self) -> bool:
        return self._reset

    @reset.setter
    def reset(self, value: bool):
        self.on_reset(value)
        self._reset = value

    @property
    def gain(self) -> Annotated[float, {'min': 0, 'max': 0.9}]:
        return self._gain

    @gain.setter
    def gain(self, value: float):
        self.set_gain(value)
        self._gain = value


gain = Gain()
