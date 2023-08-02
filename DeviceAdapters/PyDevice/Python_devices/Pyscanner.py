import nidaqmx as ni
import numpy as np
from nidaqmx.constants import TaskMode

from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogUnscaledReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter


# this is all the most recently updated functions

def scanpattern(xlims, ylims, stepsizes, padsize=0, bidirectional=False, finalpointdouble=True):
    """This produces 2 numpy arrays which can be used as input for the Galvo scanners
    xlims: [begin,end]
    ylims: [begin,end]
    stepsizes = [xsteps, ysteps]
    padsize = size of padding of first element in an xrow
    bidirectional: Boolian to make the scan back and forth alternatingly, unidirectional by default.

    Bugfix: it contains the last point twice, such that the measurement will be unscewed,

    Todo: make padding structured (sinusoid instead of flat line)
    Todo: bug fix in playrec internally
    """
    n_xsteps = int(round((xlims[1] - xlims[0]) / stepsizes[0]) + 1)
    n_ysteps = int(round((ylims[1] - ylims[0]) / stepsizes[1]) + 1)

    # This is the linear signal. Everything after is padding & structuring. Adapt here for custom patterns.
    rangex = np.linspace(xlims[0], xlims[1], n_xsteps)
    rangey = np.linspace(ylims[0], ylims[1], n_ysteps)

    xsteps = np.array([])
    ysteps = np.array([])

    for ind, y in enumerate(rangey):
        if bidirectional:
            if (ind % 2) == 0:

                xsteps = np.append(xsteps, np.append(np.ones(padsize) * rangex[0], rangex))
            else:
                xsteps = np.append(xsteps, np.append(np.ones(padsize) * rangex[-1],
                                                     np.flip(rangex)))  # flip only if index is odd.
        else:
            xsteps = np.append(xsteps, np.append(np.ones(padsize) * rangex[0], rangex))
    for y in rangey:
        ysteps = np.append(ysteps, np.ones(n_xsteps + padsize) * y)

    if finalpointdouble:
        xsteps = np.append(xsteps, xsteps[-1])
        ysteps = np.append(ysteps, ysteps[-1])

    return xsteps, ysteps


def query_devices():
    local_system = ni.system.System.local()
    driver_version = local_system.driver_version

    print('DAQmx {0}.{1}.{2}'.format(driver_version.major_version, driver_version.minor_version,
                                     driver_version.update_version))

    for device in local_system.devices:
        print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
            device.name, device.product_category, device.product_type))


def readwrite(
        outdata, sample_rate=500000, input_mapping=['Dev2/ai0'],
        output_mapping=['Dev2/ao0', 'Dev2/ao1'], input_range=[-1, 1], output_range=[-1, 1]
):
    """Function adapted from NI forum that has an electrical output signal and a electrical input signal.
    Because it goes into the dac, the input for the galvos is called output (analog out)
    and the output of the PMT is called input (analog in)

    It can handle only 1 in- and 2 outputs

    outdata: numpy array or list, will be output in analog out channel (V)

    sr: signal rate (/second), default is 500.000, the maximum of the NI USB-6341.
    Note; the NI PCIe-6363 in the lab has a maximum of 2.000.000, so this function can be overclocked.

    returns indata: measured signal from analog in channel (V)
    """
    # TODO: Make a buffer-loading & trigger function seperately
    # TODO: Make the function robust for different channel numbers

    # in order to handle both singular and multiple channel output data:
    if len(output_mapping) > 1:
        number_of_samples = outdata[0].shape[0]
    else:
        number_of_samples = outdata.shape[0]

    with ni.Task() as write_task, ni.Task() as read_task, ni.Task() as sample_clk_task:
        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.

        # We're stealing the device identifier for the clock from the input mapping string, because of backward
        # compatibility
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            f"{input_mapping[0].split('/')[0]}/ctr0", freq=sample_rate
        )
        sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=number_of_samples)

        samp_clk_terminal = f"/{input_mapping[0].split('/')[0]}/Ctr0InternalOutput"

        ao_args = {'min_val': output_range[0],
                   'max_val': output_range[1]}

        write_task.ao_channels.add_ao_voltage_chan(output_mapping[0], **ao_args)
        write_task.ao_channels.add_ao_voltage_chan(output_mapping[1], **ao_args)

        ai_args = {'min_val': input_range[0],
                   'max_val': input_range[1],
                   'terminal_config': ni.constants.TerminalConfiguration.RSE}

        write_task.timing.cfg_samp_clk_timing(
            sample_rate,
            source=samp_clk_terminal,
            active_edge=Edge.RISING,
            samps_per_chan=number_of_samples,
        )

        read_task.ai_channels.add_ai_voltage_chan(input_mapping[0], **ai_args)

        read_task.timing.cfg_samp_clk_timing(
            sample_rate,
            source=samp_clk_terminal,
            active_edge=Edge.FALLING,
            samps_per_chan=number_of_samples,
        )
        # for task in (read_task, write_task):
        #     task.timing.cfg_samp_clk_timing(rate=sample_rate, source='OnboardClock', samps_per_chan=number_of_samples)

        writer = AnalogMultiChannelWriter(write_task.out_stream)
        reader = AnalogUnscaledReader(read_task.in_stream)

        writer.write_many_sample(outdata)

        # Start the read and write tasks before starting the sample clock
        # source task.

        read_task.start()
        write_task.start()
        sample_clk_task.start()

        values_read = np.zeros([len(input_mapping), number_of_samples], dtype=np.int16)
        reader.read_int16(
            values_read, number_of_samples_per_channel=number_of_samples, timeout=30
            # the timeout determines how long we'll wait for responses.
            # (and with that determines the maximum exposure time.) Because we can imagine someone trying to make a VERY
            # long exposure image, we've set this to 30. If you use more time than that, you've probably made an implementation error
        )

    return values_read[0]


def make_voltage_zero(channels=["Dev4/ao2", "Dev4/ao3"]):
    """Puts voltage to 0, for 2 specified channels

    ToDo: unhardcode 2 channels requirement
    """

    with ni.Task() as task:
        for c in channels:
            task.ao_channels.add_ao_voltage_chan(c)

        task.write([[0], [0]], auto_start=True)  # please fix


def PMT_to_image(data,
                 x_nsteps,
                 y_nsteps,
                 x_crop,
                 y_crop,
                 depadsize=0,
                 bidirectional=False,
                 delayforward=0,
                 delaybackward=0,
                 ignore_first_point=True):
    """
    Function to write a 1-dimentional PMT signal to an image
    data: 1 dimentional numpy array

    x_nsteps: Amount of measurements taken per line in the x direction
    y_nsteps: "" in the y direction

    x_crop: floating point between 0 and 1 to determine how much the image is cropped in the x direction
    y_crop: "" in the y direction

    bidirectional: Boolian to make the reconstruction back and forth alternatingly, unidirectional by default.

    delayforward: The delay (in steps) between the electronic command and the actual system response during a left-to-right scan
    delaybackward: "" during a right-to-left scan


    ToDo: check inputs such that the length of data = x_nsteps * y_nsteps, throw error if not

    """
    if ignore_first_point:
        data = data[1:]

    # a bit crude. Shifting the data just by padding with 0s in the end and chopping away the first part
    data = np.append(data, np.zeros(delayforward))
    data = data[delayforward:]

    if bidirectional:
        full_im = np.reshape(data, [y_nsteps, x_nsteps + depadsize])

        full_im = full_im[:, depadsize:]
        full_im[1::2, :] = full_im[1::2, ::-1]  # flip all the even rows (my god this is pretty)

    else:
        full_im = np.reshape(data, [y_nsteps, x_nsteps + depadsize])
        full_im = full_im[:, depadsize:]

    return full_im

def single_capture(input_mapping=['Dev2/ai0'],
                   output_mapping=['Dev2/ao0', 'Dev2/ao1'],
                   xlims=[-1, 1],  # full range of FoV
                   ylims=[-1, 1],
                   resolution=[512, 512],
                   scanpaddingfactor=[1],
                   zoom=[1],
                   delay=[0],
                   dwelltime=[],
                   duration=[],
                   bidirectional=[True],
                   invert_values=[True],
                   input_range=[-1, 1],
                   ):
    """One single capture, standalone function that returns an image
    It can run out-of-the-box (For a full FoV scan).

    TODO: Make a buffer-loading & trigger function separately
    TODO: write this description
    """

    # check some inputs:
    if len(dwelltime) + len(duration) == 0:
        # Set to max value: 500.000 for the the NI USB-6341, 2.000.000 for the NI PCIe-6363.
        sr = 500000

    if len(dwelltime) + len(duration) > 1:
        raise Exception("Input EITHER a pixel dwelltime or a total duration")

    if bidirectional == []:
        raise Exception("Input true or false for bidirectional")

        # apply zoom factor
        # this is really bad, we need to restructure this. No redefining numbers that should be true

    xlims = [xlims[0] / zoom[0], xlims[1] / zoom[0]]
    ylims = [ylims[0] / zoom[0], ylims[1] / zoom[0]]

    #    if not stepsizes:  # if no manual stepsize was selected, calculate from required resolution
    stepx = (xlims[1] - xlims[0]) / (resolution[1] - 1)
    stepy = (ylims[1] - ylims[0]) / (resolution[0] - 1)
    stepsizes = [stepx, stepy]

    # apply padding scanning factor AFTER stepsize determination, and BEFORE n steps determination
    xlims = [xlims[0] * scanpaddingfactor[0], xlims[1] * scanpaddingfactor[0]]

    n_xsteps = int(round((xlims[1] - xlims[0]) / stepsizes[0]) + 1)
    n_ysteps = int(round((ylims[1] - ylims[0]) / stepsizes[1]) + 1)

    total_len = n_xsteps * n_ysteps
    if dwelltime:
        sr = 1 / dwelltime[0]

    if duration:
        duration = duration[0]
        sr = total_len / duration

    if sr > 2000000:
        raise Exception("Signal rate exceeds DAC maximum")

    # making scanpattern:
    xcoordinates, ycoordinates = scanpattern(xlims, ylims, stepsizes, bidirectional=bidirectional)
    sig = np.stack([xcoordinates, ycoordinates])

    # setting the DAC range:
    output_range = [min([xlims[0], ylims[0]]), max([xlims[1], ylims[1]])]

    # The DAC communication:
    indata = readwrite(sig, sr, output_mapping=output_mapping, input_mapping=input_mapping, input_range=input_range,
                       output_range=output_range)
    make_voltage_zero(output_mapping)

    # Reconstruction:
    indata = np.array(indata)
    xlims = [xlims[0] / scanpaddingfactor[0], xlims[1] / scanpaddingfactor[0]]
    x_n = int(round(((xlims[1] - xlims[0]) / stepsizes[0])) + 1)
    y_n = int(round((ylims[1] - ylims[0]) / stepsizes[1]) + 1)
    padsize = n_xsteps - x_n
    x_c = 1
    y_c = 1
    paddinginduced_delay = int(round(padsize / 2))

    image = PMT_to_image(indata, x_n, y_n, x_c, y_c, padsize, delayforward=delay[0] - paddinginduced_delay,
                         bidirectional=bidirectional)

    # A bit of data modification such that the output will always be 0 to 2^16
    if input_range[0] < 0:
        image = image + (2**16)/2

    if invert_values[0]:
        image = (2**16)-image

    return image
