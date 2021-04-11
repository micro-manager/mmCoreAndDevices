

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MP285.cpp',
	'MP285Ctrl.cpp',
	'MP285Error.cpp',
	'MP285XYStage.cpp',
	'MP285ZStage.cpp'
]


shared_library('MP285', sources, include_directories: includes, dependencies : deps, install: true)

