

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ExpCtrl.cpp',
	'FocusCallback.cpp',
	'ImgFinalProc.cpp',
	'mmgrProgRes.cpp',
	'TriggerWaitProc.cpp'
]


library('Jenoptik', sources, include_directories: includes, dependencies : deps, install: true)

