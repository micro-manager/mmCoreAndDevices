

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NikonTE2000.cpp',
	'TEHub.cpp'
]


shared_library('NikonTE2000', sources, include_directories: includes, dependencies : deps, install: true)

