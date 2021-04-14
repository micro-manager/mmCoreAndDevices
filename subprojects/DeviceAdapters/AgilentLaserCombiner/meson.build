
SDK_dep = dependency('AgilentLaserCombinerSDK', fallback: '3rdPartyPrivate', required: false)
if SDK_dep.found()
	includes = [
		include_directories('.'), # include the main directory
		]

	deps = [
		MMDevice_dep,
		SDK_dep
		]
		
	sources = [  # Must explicitly list all source files.
		'AgilentLaserCombiner.cpp'
	]


	library('AgilentLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)
else
	message('Skipping AgilentLaserCombiner DeviceAdapter. No SDK found.')
endif
