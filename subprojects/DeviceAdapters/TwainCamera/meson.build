

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CommonTWAIN.cpp',
	'dbgbox.cpp',
	'PerformanceTimer.cpp',
	'Probe.cpp',
	'ROIRectangle.cpp',
	'SaveBmpImage.cpp',
	'TwainCamera.cpp',
	'TwainDevice.cpp',
	'yyyyMMddhhmmss.cpp'
]


library('TwainCamera', sources, include_directories: includes, dependencies : deps, install: true)

