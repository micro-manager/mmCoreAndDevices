

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DeviceOperate.cpp',
	'HID.cpp',
	'hiddll.cpp',
	'Mightex.cpp',
	'Mightex_BLSDriver_SDK.cpp',
	'Mightex_LEDDriver_SDK.cpp'
]


library('Mightex_BLS', sources, include_directories: includes, dependencies : deps, install: true)

