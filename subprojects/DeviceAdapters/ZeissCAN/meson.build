

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'mcu28.cpp',
	'PhotoModule.cpp',
	'ZeissCAN.cpp',
	'ZStage.cpp'
]


shared_library('ZeissCAN', sources, dependencies : deps, install: true)

