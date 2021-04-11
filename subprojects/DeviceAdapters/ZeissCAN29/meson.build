

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ZeissCAN29.cpp',
	'ZeissHub.cpp',
	'ZeissModel.cpp',
	'ZeissMonitor.cpp'
]


shared_library('ZeissCAN29', sources, include_directories: includes, dependencies : deps, install: true)

