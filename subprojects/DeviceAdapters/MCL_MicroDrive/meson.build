

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('MCL_MicroDrive', sources, dependencies : deps, install: true)

