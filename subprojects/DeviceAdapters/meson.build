project('DeviceAdapters', 'cpp', version : '0.0.1')

MMDevice_dep = dependency('MMDevice', fallback: ['MMDevice', 'MMDevice_dep'])

pymod = import('python')  # Import mesons python module
python = pymod.find_installation('python3')  # Get a reference to python
cmd = run_command(python, 'getSubdirs.py')  # run a python script.
subdirs = cmd.stdout().strip().split('\n')  # Parse the scripts output to get the subdirs.
fs = import('fs')  # Get the filesystem module
foreach subpath : subdirs
	if fs.exists(join_paths(subpath, 'meson.build'))
		subdir(subpath) ## Use the build script defined in the subdir
	else  # Use the default procedure.
		message('Skipping @0@'.format(subpath))
	endif
endforeach
