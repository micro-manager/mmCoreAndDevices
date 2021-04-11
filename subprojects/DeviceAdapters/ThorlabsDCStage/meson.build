

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ThorlabsDCStage.cpp'
]


shared_library('ThorlabsDCStage', sources, include_directories: includes, dependencies : deps, install: true)

