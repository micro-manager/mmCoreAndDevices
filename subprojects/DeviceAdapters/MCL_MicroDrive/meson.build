

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'HandleListType.cpp',
	'handle_list_if.cpp',
	'MCL_MicroDrive.cpp',
	'mdutils.cpp',
	'MicroDriveXYStage.cpp',
	'MicroDriveZStage.cpp'
]


library('MCL_MicroDrive', sources, include_directories: includes, dependencies : deps, install: true)

