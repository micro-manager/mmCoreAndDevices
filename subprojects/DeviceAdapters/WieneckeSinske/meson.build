

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CAN29.cpp',
	'WieneckeSinske.cpp'
]


shared_library('WieneckeSinske', sources, include_directories: includes, dependencies : deps, install: true)

