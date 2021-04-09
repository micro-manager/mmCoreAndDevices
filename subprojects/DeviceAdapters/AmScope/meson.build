

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AmScope.cpp',
	'SequenceThread.cpp'
]


shared_library('AmScope', sources, dependencies : deps, install: true)

