

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'error_code.cpp',
	'module.cpp',
	'TCPIPPort.cpp',
	'Util.cpp'
]


shared_library('TCPIPPort', sources, include_directories: includes, dependencies : deps, install: true)

