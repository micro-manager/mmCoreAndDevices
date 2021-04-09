project('DeviceAdapters', 'cpp', version : '0.0.1')

MMDevice_dep = dependency('MMDevice', fallback: ['MMDevice', 'MMDevice_dep'])
boost_dep = dependency('boost', modules: ['chrono', 'date_time', 'regex', 'system', 'thread'])

add_project_arguments('-DMODULE_EXPORTS', language: ['cpp'])

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', '-D_WINDOWS', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-D_NDEBUG', language : ['cpp'])
else
    add_project_arguments('-D_DEBUG', language : ['cpp'])
endif


pymod = import('python')  # Import mesons python module
python = pymod.find_installation('python3')  # Get a reference to python
cmd = run_command(python, 'getSubdirs.py')  # run a python script.
subdirs = cmd.stdout().strip().split('\n')  # Parse the scripts output to get the subdirs.
fs = import('fs')  # Get the filesystem module
foreach subpath : subdirs
	if fs.exists(join_paths(subpath, 'meson.build'))
		subdir(subpath) ## Use the build script defined in the subdir
	else  # Use the default procedure.
		message('Default config: @0@'.format(subpath))
		includes = include_directories(subpath)
		deps = [boost_dep, MMDevice_dep]
		cmd = run_command(python, 'getSourcefiles.py', subpath)
		sources = cmd.stdout().strip().split('\n')
		if sources[0] == ''
			message('No Sources found')
			continue
		endif
		shared_library(subpath, sources, include_directories: includes, dependencies: deps, install: true, required: false)
	endif
endforeach
