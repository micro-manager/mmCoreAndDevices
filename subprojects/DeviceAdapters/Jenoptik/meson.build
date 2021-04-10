

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ExpCtrl.cpp',
	'FocusCallback.cpp',
	'ImgFinalProc.cpp',
	'mmgrProgRes.cpp',
	'TriggerWaitProc.cpp'
]


shared_library('Jenoptik', sources, include_directories: includes, dependencies : deps, install: true)

