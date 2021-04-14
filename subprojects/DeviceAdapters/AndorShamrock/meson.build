SDK_dep = dependency('AndorShamrockSDK', fallback: '3rdPartyPrivate', required: false)

if SDK_dep.found()

	includes = [
		include_directories('.'), # include the main directory
		]

	deps = [
		MMDevice_dep,
		SDK_dep
	]
		
	sources = [  # Must explicitly list all source files.
		'AndorShamrock.cpp',
		'ModuleAPIFunctions.cpp',
		'ShamrockConstants.cpp'
	]


	library('AndorShamrock', sources, include_directories: includes, dependencies : deps, install: true)
else
	message('Skipping AndorShamrock DeviceAdapter. No SDK found.')
endif
