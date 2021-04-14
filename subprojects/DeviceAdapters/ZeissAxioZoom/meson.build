

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DevicesX.cpp',
	'ZeissAxioZoom.cpp',
	'ZeissHub.cpp',
	'ZeissModelX.cpp',
	'ZeissMonitor.cpp'
]


library('ZeissAxioZoom', sources, include_directories: includes, dependencies : deps, install: true)

