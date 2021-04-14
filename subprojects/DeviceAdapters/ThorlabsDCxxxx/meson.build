

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DC2200.cpp',
	'DC2XXX.cpp',
	'DC3100.cpp',
	'DC4100.cpp',
	'DCxxxx_Plugin.cpp',
	'DynError.cpp'
]


library('ThorlabsDCxxxx', sources, include_directories: includes, dependencies : deps, install: true)

