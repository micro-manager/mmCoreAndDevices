

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('IIDC', sources, dependencies : deps, install: true)

