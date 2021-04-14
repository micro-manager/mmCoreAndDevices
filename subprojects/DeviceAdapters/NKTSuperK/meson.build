

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SuperKDevice.cpp',
	'SuperKExtreme.cpp',
	'SuperKHub.cpp',
	'SuperKModule.cpp',
	'Varia.cpp'
]


library('NKTSuperK', sources, include_directories: includes, dependencies : deps, install: true)

