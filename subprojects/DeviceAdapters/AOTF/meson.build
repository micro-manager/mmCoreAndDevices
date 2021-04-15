inpout_dep = dependency('inpout', fallback: '3rdPartyPrivate', required: false)

if inpout_dep.found()

	includes = [
		include_directories('.') # include the main directory
		]

	deps = [
		MMDevice_dep,
		inpout_dep
		]
		
	sources = [  # Must explicitly list all source files.
		'AOTF.cpp'
	]


	library('AOTF', sources, include_directories: includes, dependencies : deps, install: true)
else
	message('Skipping AOTF DeviceAdapter. Missing dependency')
endif