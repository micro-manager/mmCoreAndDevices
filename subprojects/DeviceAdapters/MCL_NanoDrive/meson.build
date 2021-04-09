

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
	'MCL_NanoDrive.cpp',
	'MCL_NanoDrive_XYStage.cpp',
	'MCL_NanoDrive_ZStage.cpp'
]


shared_library('MCL_NanoDrive', sources, dependencies : deps, install: true)

