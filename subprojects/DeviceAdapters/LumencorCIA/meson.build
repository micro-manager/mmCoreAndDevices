

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CIA.cpp',
	'LECtrl.cpp'
]


library('LumencorCIA', sources, include_directories: includes, dependencies : deps, install: true)

