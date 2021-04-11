SDK_dir = join_paths(MM_3RDPARTY_PRIVATE, 'ShamrockSDK')

includes = [
    include_directories('.'), # include the main directory
	include_directories(join_paths(SDK_dir, 'include'))
    ]

deps = [
	MMDevice_dep,
	meson.get_compiler('cpp').find_library('ShamrockCIFm', dirs: [join_paths(SDK_dir, 'lib', 'x64')], static: true)
	]
	
sources = [  # Must explicitly list all source files.
	'AndorShamrock.cpp',
	'ModuleAPIFunctions.cpp',
	'ShamrockConstants.cpp'
]


shared_library('AndorShamrock', sources, include_directories: includes, dependencies : deps, install: true)

