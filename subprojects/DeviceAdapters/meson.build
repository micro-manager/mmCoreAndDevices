project('DeviceAdapters', 'cpp', version : '0.0.1',
	default_options: [
		'default_library=shared'  # Build DLLs rather than Libs
	]
)

MMDevice_dep = dependency('MMDevice', fallback: ['MMDevice'])

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

getSubDirsScript = '''  # Indentation matters here
from glob import glob
from os import path
wDir = r"@0@"
files = [f[len(wDir)+1:] for f in glob(path.join(wDir, '@1@')) if path.isdir(f)]
print(','.join(files))
'''.format(meson.current_source_dir(), get_option('SearchPattern'))

cmd = run_command(python, '-c', getSubDirsScript)  # run a python script.
subdirs = cmd.stdout().strip().split(',')  # Parse the scripts output to get the subdirs.
fs = import('fs')  # Get the filesystem module
foreach subpath : subdirs
	if fs.exists(join_paths(subpath, 'meson.build'))
		subdir(subpath) ## Use the build script defined in the subdir
	else  # Skip if no build file.
		continue
	endif
endforeach
