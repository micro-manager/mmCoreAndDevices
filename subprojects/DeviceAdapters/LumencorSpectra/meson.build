

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Spectra.cpp'
]


shared_library('LumencorSpectra', sources, dependencies : deps, install: true)

