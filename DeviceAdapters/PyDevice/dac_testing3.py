import random
import numpy

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogUnscaledReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

#######################################################################################################################
# This entire script was adapted from multiple test & example scripts in the Nidaqmx GitHub repo.

number_of_samples = 500000
sample_rate = 250000
# this cannot surpass 500.000/ n_channels; because that violates the maximum sample speed of the NI USB-6341 that we're using

number_of_channels = 2
dev_name = 'Dev2'  # < remember to change to your device name, and channel input names below.
ao0 = '/ao0'
ao1 = '/ao1'
ai0 = '/ai0'
ai1 = '/ai1'



with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task, nidaqmx.Task() as sample_clk_task:
    # Use a counter output pulse train task as the sample clock source
    # for both the AI and AO tasks.

    sample_clk_task.co_channels.add_co_pulse_chan_freq(
        f"{dev_name}/ctr0", freq=sample_rate
    )
    sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=number_of_samples)

    ao_args = {'min_val': -1,
               'max_val': 1}

    write_task.ao_channels.add_ao_voltage_chan(dev_name+ao0, **ao_args)
    write_task.ao_channels.add_ao_voltage_chan(dev_name+ao1, **ao_args)

    ai_args = {'min_val': -1,
               'max_val': 1,
               'terminal_config': nidaqmx.constants.TerminalConfiguration.RSE}

    read_task.ai_channels.add_ai_voltage_chan(dev_name + ai0, **ai_args)
    read_task.ai_channels.add_ai_voltage_chan(dev_name + ai1, **ai_args)

    write_task.timing.cfg_samp_clk_timing(
        sample_rate,
        active_edge=Edge.RISING,
        samps_per_chan=number_of_samples,
    )

    read_task.timing.cfg_samp_clk_timing(
        sample_rate,
        active_edge=Edge.FALLING,
        samps_per_chan=number_of_samples,
    )

    writer = AnalogMultiChannelWriter(write_task.out_stream)
    reader = AnalogUnscaledReader(read_task.in_stream)

    values_to_test = numpy.array([np.arange(-1,1,0.000004),np.arange(-1,1,0.000004)])
    writer.write_many_sample(values_to_test)

    # Start the read and write tasks before starting the sample clock
    # source task.
    read_task.start()
    write_task.start()
    sample_clk_task.start()

    values_read = numpy.zeros((number_of_channels, number_of_samples), dtype=numpy.int16)
    reader.read_int16(
        values_read, number_of_samples_per_channel=number_of_samples, timeout=30 # the timeout determines how long we'll wait for responses.
        # (and with that determines the maximum exposure time.) Because we can imagine someone trying to make a VERY
        # long exposure image, we've set this to 30. If you use more time than that, you've probably made an implementation error
    )
    import matplotlib.pyplot as plt
    plt.plot(values_read[0])
    plt.show()