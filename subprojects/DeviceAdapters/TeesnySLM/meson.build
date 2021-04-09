

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'TeensyShutter.cpp',
	'TeensySLM.cpp'
]


shared_library('TeesnySLM', sources, dependencies : deps, install: true)

