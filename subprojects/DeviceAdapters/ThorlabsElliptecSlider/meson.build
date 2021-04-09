

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ThorlabsElliptecSlider.cpp'
]


shared_library('ThorlabsElliptecSlider', sources, dependencies : deps, install: true)

