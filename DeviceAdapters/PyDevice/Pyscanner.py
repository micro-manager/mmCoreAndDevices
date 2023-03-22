import nidaqmx as ni
import numpy as np
from nidaqmx.constants import TaskMode
import pylab as plt
import ast


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


def playrec(
        outdata, sr=500000, input_mapping=['Dev2/ai0'],
        output_mapping=['Dev2/ao0'], input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]
):
    """Function adapted from NI forum that has an electrical output signal and a electrical input signal.
    Because it goes into the dac, the input for the galvos is called output (analog out)
    and the output of the PMT is called input (analog in)

    It can handle multiple in- and outputs, default is 1 each.

    outdata: numpy array or list, will be output in analog out channel (V)

    sr: signal rate (/second), default is 500.000, the maximum of the NI USB-6341.
    Note; the NI PCIe-6363 in the lab has a maximum of 2.000.000, so this function can be overclocked.

    returns indata: measured signal from analog in channel (V)
    """

    # in order to handle both singular and multiple channel output data:
    if len(output_mapping) > 1:
        nsamples = outdata[0].shape[0]
    else:
        nsamples = outdata.shape[0]

    with ni.Task() as read_task, ni.Task() as write_task:
        for o in output_mapping:
            aochan = write_task.ao_channels.add_ao_voltage_chan(o)
            aochan.ao_min = output_range[0]
            aochan.ao_max = output_range[1]  # output range
            
        for i in input_mapping:
            aichan = read_task.ai_channels.add_ai_voltage_chan(i)
            aichan.ai_min = input_range[0]
            aichan.ai_max = input_range[1]

        for task in (read_task, write_task):
            task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=nsamples)

        write_task.triggers.start_trigger.cfg_dig_edge_start_trig(read_task.triggers.start_trigger.term)
        write_task.write(outdata, auto_start=True)
        indata = read_task.read(nsamples)
    return indata


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


    ToDo: work in delay functionality
    ToDo: work in cropping functionality
    ToDo: check inputs such that the length of data = x_nsteps * y_nsteps, throw error if not
    ToDo: work the rectification into the playrec function
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


def stringinterpret(input_list):
    input_list = [str(x) for x in input_list]


def single_capture(input_mapping=['Dev2/ai0'],
                   output_mapping=['Dev2/ao0', 'Dev2/ao1'],
                   xlims=[-0.5, 0.5],  # full range of FoV
                   ylims=[-0.5, 0.5],
                   resolution=[512, 512],
                   scanpaddingfactor=[1],
                   zoom=[1],
                   delay=[0],
                   dwelltime=[],
                   duration=[],
                   bidirectional=[True],
                   input_range=[-1.5,1.5],
                   ):
    """One single capture, standalone function that returns an image
    It can run out-of-the-box (For a full FoV scan).


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
    stepx = (xlims[1] - xlims[0]) / (resolution[0] - 1)
    stepy = (ylims[1] - ylims[0]) / (resolution[1] - 1)
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
    output_range = [min([xlims[0],ylims[0]]),max([xlims[1],ylims[1]])]

    # The DAC communication:
    indata = playrec(sig, sr, output_mapping=output_mapping, input_mapping=input_mapping,input_range=input_range, output_range=output_range)
    make_voltage_zero(output_mapping)
    
    # Reconstruction:
    indata = np.array(indata)
    xlims = [xlims[0] / scanpaddingfactor[0], xlims[1] / scanpaddingfactor[0]]
    x_n = int(round(((xlims[1] - xlims[0]) / stepsizes[0])) + 1)
    y_n = int(round((ylims[1] - ylims[0]) / stepsizes[1]) + 1)
    padsize = n_xsteps - x_n
    x_c = 1
    y_c = 1
    paddinginduced_delay = int(round(padsize/2))
    
    image = PMT_to_image(indata, x_n, y_n, x_c, y_c, padsize, delayforward=delay[0]-paddinginduced_delay, bidirectional=bidirectional)
    #    image = np.round((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype(int)
    # image = np.round((image - image.min()) * (np.iinfo(np.int16).max / (image.max() - image.min()))).astype(np.int16)
    #image = (image - image.min()) * (1 / (image.max() - image.min()))  # hopefully this scales it from 0 to 1
    # hopefully this is now redundant
    return image


def cpp_single_capture(input_mapping_str, output_mapping_str, resolution_str, zoom_str, delay_str, dwelltime_str,scanpadding_str,input_range_str):
    """Function written to connect the cpp call to the function. This is because c++ calls functions with the same
    data structure (uint8 strings). This could be fixed in c++, but this was considered clearer.
    If the call is changed in the device adapter, it should be changed here too.
    Inputs: uint8 strings
    Outputs: other data types
    """
    # change the type of the c++ inputs
    input_mapping_str = ast.literal_eval(input_mapping_str)
    output_mapping_str = ast.literal_eval(output_mapping_str)
    resolution_str = np.array(ast.literal_eval(resolution_str), dtype=int)
    zoom_str = np.array(ast.literal_eval(zoom_str), dtype=int)
    delay_str = np.array(ast.literal_eval(delay_str), dtype=int)
    dwelltime_str = np.array(ast.literal_eval(dwelltime_str), dtype=float)
    scanpadding_str = np.array(ast.literal_eval(scanpadding_str), dtype=float)
    input_range_str = np.array(ast.literal_eval(input_range_str), dtype=float)

    image = single_capture(input_mapping=input_mapping_str,
        output_mapping=output_mapping_str,
        resolution=resolution_str,
        zoom=zoom_str,
        delay=delay_str,
        dwelltime=dwelltime_str,
        scanpaddingfactor = scanpadding_str,
        input_range = input_range_str
    )

    return image