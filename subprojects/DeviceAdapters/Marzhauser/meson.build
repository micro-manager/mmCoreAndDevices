

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Marzhauser.cpp'
]


shared_library('Marzhauser', sources, dependencies : deps, install: true)

