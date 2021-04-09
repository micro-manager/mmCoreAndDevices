

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'KuriosLCTF.cpp',
	'KuriosLCTFModule.cpp'
]


shared_library('KuriosLCTF', sources, dependencies : deps, install: true)

