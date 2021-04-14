

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'HandleListType.cpp',
	'handle_list_if.cpp',
	'MCL_NanoDrive.cpp',
	'MCL_NanoDrive_XYStage.cpp',
	'MCL_NanoDrive_ZStage.cpp'
]


library('MCL_NanoDrive', sources, include_directories: includes, dependencies : deps, install: true)

