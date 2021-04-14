

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Controller.cpp',
	'PIGCSControllerCom.cpp',
	'PIGCSControllerDLL.cpp',
	'PIXYStage_DLL.cpp',
	'PIZStage_DLL.cpp',
	'PI_GCS_2.cpp'
]


library('PI_GCS_2', sources, include_directories: includes, dependencies : deps, install: true)

