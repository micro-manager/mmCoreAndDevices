

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XLight.cpp',
	'XLIGHTHub.cpp'
]


shared_library('XLight', sources, dependencies : deps, install: true)

