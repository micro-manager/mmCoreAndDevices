

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'KuriosLCTF.cpp',
	'KuriosLCTFModule.cpp'
]


library('KuriosLCTF', sources, include_directories: includes, dependencies : deps, install: true)

