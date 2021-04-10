

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AndorSDK3.cpp',
	'AOIProperty.cpp',
	'BooleanProperty.cpp',
	'BooleanPropertyWithPoiseControl.cpp',
	'CallBackManager.cpp',
	'EnumProperty.cpp',
	'EventsManager.cpp',
	'ExposureProperty.cpp',
	'FloatProperty.cpp',
	'IntegerProperty.cpp',
	'pevents.cpp',
	'SnapShotControl.cpp',
	'SRRFAndorSDK3Camera.cpp',
	'stdafx.cpp'
]


shared_library('AndorSDK3', sources, include_directories: includes, dependencies : deps, install: true)

