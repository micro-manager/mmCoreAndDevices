

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CAN29.cpp',
	'WieneckeSinske.cpp'
]


shared_library('WieneckeSinske', sources, dependencies : deps, install: true)

