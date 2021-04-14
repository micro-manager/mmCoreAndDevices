

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'OnBoardHW.cpp',
	'Oxxius_combiner.cpp'
]


library('OxxiusCombiner', sources, include_directories: includes, dependencies : deps, install: true)

