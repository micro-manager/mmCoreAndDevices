

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CIA.cpp',
	'LECtrl.cpp'
]


shared_library('LumencorCIA', sources, include_directories: includes, dependencies : deps, install: true)

