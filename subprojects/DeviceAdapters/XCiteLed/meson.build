

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XLed.cpp',
	'XLedCtrl.cpp',
	'XLedDev.cpp'
]


shared_library('XCiteLed', sources, include_directories: includes, dependencies : deps, install: true)

