

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ZeissCAN29.cpp',
	'ZeissHub.cpp',
	'ZeissModel.cpp',
	'ZeissMonitor.cpp'
]


shared_library('ZeissCAN29', sources, dependencies : deps, install: true)

