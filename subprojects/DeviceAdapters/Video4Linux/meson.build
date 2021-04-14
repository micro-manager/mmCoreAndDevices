

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'video4linux2.cpp'
]


library('Video4Linux', sources, include_directories: includes, dependencies : deps, install: true)

