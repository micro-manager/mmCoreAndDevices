import sys
import os

path = os.path.split(sys.executable)[0]

properties = r"""<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <PropertyGroup>
    <IncludePath>$(python_path)\include;$(python_path)\Lib\site-packages\numpy\core\include;$(IncludePath)</IncludePath>
    <LibraryPath>$(python_path)\libs;$(LibraryPath)</LibraryPath>
  </PropertyGroup>
</Project>""".replace("$(python_path)", path)

print(properties)
