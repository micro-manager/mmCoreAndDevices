

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'OnBoardHW.cpp',
	'Oxxius_combiner.cpp'
]


shared_library('OxxiusCombiner', sources, include_directories: includes, dependencies : deps, install: true)

