

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'mcu28.cpp',
	'PhotoModule.cpp',
	'ZeissCAN.cpp',
	'ZStage.cpp'
]


library('ZeissCAN', sources, include_directories: includes, dependencies : deps, install: true)

