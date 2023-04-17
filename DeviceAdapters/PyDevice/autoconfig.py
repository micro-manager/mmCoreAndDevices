import sys
import os
import numpy

os.system('git update-index --assume-unchanged AutoConfig.props')  # we don't want git to track changes to this file, but an empty template should be in the repository or VS will not open the Project
path = os.path.split(sys.executable)[0]
mmpath = os.path.join(os.environ["ProgramFiles"], "Micro-Manager-2.0", "mmgr_dal_PyDevice.dll")
nppath = numpy.__path__[0]
# check if the folder exists and if we can write to it
#try:
#    tmpfile = mmpath + '.tmp'
#    open(tmpfile, 'w').close()
#    os.remove(tmpfile)
#    install_command = f'copy "$(TargetPath)" "{mmpath}"'
#except:
#    install_command = ''
install_command = ''

properties = r"""<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <PropertyGroup>
    <IncludePath>$(python_path)\include;$(numpy_path)\core\include;$(IncludePath)</IncludePath>
    <LibraryPath>$(python_path)\libs;$(LibraryPath)</LibraryPath>
  </PropertyGroup>
  <ItemDefinitionGroup>
    <PostBuildEvent>
      <Command>$(install_command)</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
</Project>""".replace("$(python_path)", path).replace("$(install_command)", install_command)\
    .replace("$(numpy_path)", nppath).replace("$(install_command)", install_command)

print(properties)




