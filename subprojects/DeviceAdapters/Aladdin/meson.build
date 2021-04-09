

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Aladdin.cpp'
]


shared_library('Aladdin', sources, dependencies : deps, install: true)

