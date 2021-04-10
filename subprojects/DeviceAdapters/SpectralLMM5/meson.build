

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SpectralLMM5.cpp',
	'SpectralLMM5Interface.cpp'
]


shared_library('SpectralLMM5', sources, include_directories: includes, dependencies : deps, install: true)

