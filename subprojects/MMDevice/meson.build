project('MMDevice', 'cpp', version : '0.0.1')

mm_version = meson.project_version()
ver_arr = mm_version.split('.')

as_major_version = ver_arr[0]
as_minor_version = ver_arr[1]
as_micro_version = ver_arr[2]

add_project_arguments('-DWIN32', '-DDEBUG', '-D_LIB', '-DMODULE_EXPORTS', language: 'cpp')


includes = include_directories('.') # include the config.h file.
sources = [
	'MMDevice.cpp',
	'Debayer.cpp',
	'DeviceUtils.cpp',
	'ImgBuffer.cpp', 
	'ModuleInterface.cpp', 
	'Property.cpp'
]

MMDev_lib = static_library('MMDevice', sources, include_directories: includes)

MMDev_dep = declare_dependency(include_directories: includes, link_with: MMDev_lib) 