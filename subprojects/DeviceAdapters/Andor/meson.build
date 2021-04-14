

includes = [
    include_directories('.'), # include the main directory
    ]

deps = [
	dependency('boost', modules: ['filesystem']),
	MMDevice_dep,
	dependency('AndorSDK_v2.95', fallback: '3rdPartyPrivate')
	]
	
sources = [  # Must explicitly list all source files.
	'Andor.cpp',
	'ReadModeControl.cpp',
	'SpuriousNoiseFilterControl.cpp',
	'SRRFAndorCamera.cpp',
	'SRRFControl.cpp'
]


library('Andor', sources, include_directories: includes, dependencies : deps, install: true)

