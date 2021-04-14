
SDK_dir = join_paths(MM_3RDPARTY_PRIVATE, 'Andor', 'SDK 2.95.30003.0')
SDK_dep = meson.get_compiler('cpp').find_library('atmcd64m', dirs: [SDK_dir], static: true)

includes = [
    include_directories('.'), # include the main directory
	include_directories(SDK_dir)
    ]

deps = [
	dependency('boost', modules: ['filesystem']),
	MMDevice_dep,
	SDK_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Andor.cpp',
	'ReadModeControl.cpp',
	'SpuriousNoiseFilterControl.cpp',
	'SRRFAndorCamera.cpp',
	'SRRFControl.cpp'
]


library('Andor', sources, include_directories: includes, dependencies : deps, install: true)

