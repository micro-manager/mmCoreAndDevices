SDK_dep = dependency('JAI_SDK_v1.4.1', fallback: '3rdPartyPrivate', required: false)

if SDK_dep.found()
	includes = [
		include_directories('.') # include the main directory
		]

	deps = [
		MMDevice_dep,
		SDK_dep
		]
		
	sources = [  # Must explicitly list all source files.
		'GigECamera.cpp',
		'GigECameraAcq.cpp',
		'GigECameraAction.cpp',
		'GigENodes.cpp'
	]

	library('GigECamera', sources, include_directories: includes, dependencies : deps, install: true)
else
	message('Skipping GigECamera DeviceAdapter. No SDK found')
endif

