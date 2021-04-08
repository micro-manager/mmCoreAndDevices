project('MMCore', 'cpp', version : '0.0.1')

mm_version = meson.project_version()
ver_arr = mm_version.split('.')

as_major_version = ver_arr[0]
as_minor_version = ver_arr[1]
as_micro_version = ver_arr[2]

conf = configuration_data({
	'WIN32': true,
	'DEBUG': true,
	'_LIB': true,
	'_WINDOWS': true
	})
	
configure_file( output : 'projconfig.h',
               configuration : conf)
config_include = include_directories('.') # include the config.h file.
dev_inc = include_directories('../MMDevice')

deps = [
	dependency('boost', required: true)
	]


static_library('MMCore', 'MMCore.cpp', include_directories: [config_include, dev_inc], dependencies : deps)