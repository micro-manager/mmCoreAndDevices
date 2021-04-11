

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'IIDCCamera.cpp',
	'IIDCCapture.cpp',
	'IIDCFeature.cpp',
	'IIDCInterface.cpp',
	'IIDCUtils.cpp',
	'IIDCVendorAVT.cpp',
	'IIDCVideoMode.cpp',
	'MMIIDCCamera.cpp'
]


shared_library('IIDC', sources, include_directories: includes, dependencies : deps, install: true)

