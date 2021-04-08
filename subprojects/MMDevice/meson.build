project('MMDevice', 'cpp', version : '0.0.1')

mm_version = meson.project_version()
ver_arr = mm_version.split('.')

as_major_version = ver_arr[0]
as_minor_version = ver_arr[1]
as_micro_version = ver_arr[2]

conf = configuration_data({
	'WIN32': true,
	'DEBUG': true,
	'_LIB': true,
	'MODULE_EXPORTS': true
	})
	
configure_file( output : 'config.h',
               configuration : conf)
config_include = include_directories('.') # include the config.h file.


MMDev_lib = static_library('MMDevice', 'MMDevice.cpp', include_directories: config_include)

MMDev_dep = declare_dependency(include_directories: config_include, link_with: MMDev_lib) 