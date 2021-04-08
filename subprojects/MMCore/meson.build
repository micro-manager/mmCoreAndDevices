project('MMCore', 'cpp', version : '0.0.1')

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
	dependency('boost', required: true),
	]

subproject('MMDevice')

MMCore_lib = static_library('MMCore', 'MMCore.cpp', include_directories: [config_include, dev_inc], dependencies : deps)
MMCore_dep = declare_dependency(include_directories: config_include, link_with: MMCore_lib)