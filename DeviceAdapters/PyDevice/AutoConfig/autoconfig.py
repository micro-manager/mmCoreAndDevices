import sys
from os import path, environ, system
import numpy


# Current directory
project_path = path.dirname(path.dirname(__file__))


def locate(p, description, should_have):
    if not path.isdir(p) or not path.isfile(path.join(p, should_have)):
        print(f"Could not find {description} in folder {p}, please configure the path manually in AutoConfig.props.")
    return fix_path(p, description)

def fix_path(p, description): 
    p = path.join(path.normpath(p), "")
    print(f"Located {description} in: {p}")
    return p

def rewrite(file):  # construct file from template
    with open(path.join(project_path, "AutoConfig", file + ".template")) as template:
        config = template.read()
        config = config.replace('@python_path@', python_path)
        config = config.replace('@numpy_path@', numpy_path)
        config = config.replace('@mmexe_path@', mm_exe_path)
        config = config.replace('@mmsrc_path@', mm_src_path)

    with open(path.join(project_path, file), "w") as config_file:
        config_file.write(config)

         

# Locate Micro-Manager-2.0 executable
mm_exe_path = path.join(environ["ProgramFiles"], "Micro-Manager-2.0", "")
mm_exe_path = locate(mm_exe_path, "the Micro-Manager executable", "ImageJ.exe")

# Locate Micro-Manager source code
mm_src_path = path.join(path.dirname(project_path), "mmCoreAndDevices")
if not path.isdir(mm_src_path):
    mm_src_path = path.dirname(project_path)

mm_src_path = locate(mm_src_path, "the mmCoreAndDevices repository", "micromanager.sln")
# add option to automatically check out repository in the correct location?

# Locate Python sdk base directory and nupy sdk
python_path = fix_path(path.split(sys.executable)[0], "the Python SDK")
numpy_path = fix_path(numpy.__path__[0], "the NumPy SDK")

# rewrite sln template

rewrite('AutoConfig.props')
rewrite('PyDevice.sln')

# we don't want git to track changes to this file, but an empty template should be in the repository or VS will not open the Project
system(f"git update-index --assume-unchanged {path.join(project_path, 'AutoConfig', 'AutoConfig.props')}")
system(f"git update-index --assume-unchanged {path.join(project_path, 'PyDevice.sln')}")

