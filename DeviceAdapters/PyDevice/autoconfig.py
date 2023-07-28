import sys
from os import path, environ
import numpy

def locate(p, description, should_have):
    while not path.isdir(p) or not path.isfile(path.join(p, should_have)):
        p = input(
            f"Could not find {description} in folder {p}, please type the path to the folder "
            f"containing the executable, or press Enter to skip.")
        if not p:
            print("You can manually configure the path later in the AutoConfig.props file.")
            return
        if path.split(p)[-1].endswith('exe'):
            p = path.split(pp)[0]
        if not p.endswith('\\') and not p.endswith('/'):
            p = path.join(p, "")

    print(f"{description}: {p}")
    return p

def rewrite(file):  # construct file from template
    with open(file + ".template") as template:
        config = template.read()
        config = config.replace('@python_path@', python_path)
        config = config.replace('@numpy_path@', numpy_path)
        config = config.replace('@mmexe_path@', mm_exe_path)
        config = config.replace('@mmsrc_path@', mm_src_path)

    with open(file, "w") as config_file:
        config_file.write(config)


# Current directory
project_path = path.dirname(__file__)

# Locate Micro-Manager-2.0 executable
mm_exe_path = path.join(environ["ProgramFiles"], "Micro-Manager-2.0", "")
mm_exe_path = locate(mm_exe_path, "the Micro-Manager executable", "ImageJ.exe")

# Locate Micro-Manager source code
mm_src_path = path.join(path.dirname(project_path), "mmCoreAndDevices")
if not path.isdir(mm_src_path):
    mm_src_path = path.dirname(project_path)

mm_src_path = locate(mm_src_path, "the mmCoreAndDevices repository", "micromanager.sln")
# add option to automatically check out repository in the correct location?

# Locate Python sdk base directory
python_path = path.split(sys.executable)[0]

# Locate numpy sdk
numpy_path = numpy.__path__[0]

# rewrite sln template
rewrite(path.join(project_path, 'AutoConfig.props'))
rewrite(path.join(project_path, 'PyDevice.sln'))

