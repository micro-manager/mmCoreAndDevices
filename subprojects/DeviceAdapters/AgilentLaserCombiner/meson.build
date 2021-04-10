



SDK_dir = join_paths(MM_3RDPARTY_PRIVATE, 'Agilent', 'LaserCombiner')
SDK_dep = meson.get_compiler('cpp').find_library('LaserCombinerSDK64', 
												dirs: SDK_dir, 
												static: true												)
SDK_inc = include_directories(SDK_dir)

includes = [
    include_directories('.'), # include the main directory
	SDK_inc
    ]

deps = [
	boost_dep,
	MMDevice_dep,
	SDK_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AgilentLaserCombiner.cpp'
]

install_data('lib/LaserCombinerSDK64.dll')

shared_library('AgilentLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)

