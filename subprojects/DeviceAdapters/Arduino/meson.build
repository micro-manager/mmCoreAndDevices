

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Arduino.cpp'
]


library('Arduino', sources, include_directories: includes, dependencies : deps, install: true)

