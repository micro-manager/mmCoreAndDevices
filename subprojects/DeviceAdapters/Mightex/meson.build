

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DeviceOperate.cpp',
	'HID.cpp',
	'hiddll.cpp',
	'Mightex.cpp',
	'Mightex_LEDDriver_SDK.cpp'
]


shared_library('Mightex', sources, include_directories: includes, dependencies : deps, install: true)

