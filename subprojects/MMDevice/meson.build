project('MMDevice', 'cpp', version : '0.0.1')

mm_version = meson.project_version()
ver_arr = mm_version.split('.')

as_major_version = ver_arr[0]
as_minor_version = ver_arr[1]
as_micro_version = ver_arr[2]

add_project_arguments('-D_LIB', '-DMODULE_EXPORTS', language: 'cpp')

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-DNDEBUG', language : ['cpp'])
else
    add_project_arguments('-DDEBUG', language : ['cpp'])
endif


includes = include_directories('.') # include the config.h file.
sources = [
	'MMDevice.cpp',
	'Debayer.cpp',
	'DeviceUtils.cpp',
	'ImgBuffer.cpp', 
	'ModuleInterface.cpp', 
	'Property.cpp'
]

MMDevice_lib = static_library('MMDevice', sources, include_directories: includes)

MMDevice_dep = declare_dependency(include_directories: includes, link_with: MMDevice_lib) 