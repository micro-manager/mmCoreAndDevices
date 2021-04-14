
SDK_dep = dependency('AndorLaserCombinerSDK', fallback:'3rdPartyPrivate', required: false)

if SDK_dep.found()

	includes = [
		include_directories('.'), # include the main directory
		]

	deps = [
		MMDevice_dep,
		dependency('boost'),
		SDK_dep
		]
		
	sources = [  # Must explicitly list all source files.
		'AndorLaserCombiner.cpp'
	]


	library('AndorLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)
else
	message('Skipping AndorLaserCombiner DeviceAdapter. No SDK found.')
endif

