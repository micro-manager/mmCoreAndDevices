

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('Thorlabs', sources, dependencies : deps, install: true)

