

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'IntegratedFilterWheel.cpp',
	'MotorStage.cpp',
	'MotorZStage.cpp',
	'PiezoZStage.cpp',
	'Thorlabs.cpp',
	'XYStage.cpp'
]


shared_library('Thorlabs', sources, include_directories: includes, dependencies : deps, install: true)

