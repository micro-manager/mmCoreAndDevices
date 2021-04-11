# This project was too messy to include in the new meson build system. It was including files from other projects.

#SDK_dep = meson.get_compiler('cpp').find_library('atcorem', dirs: [join_paths(meson.current_source_dir(), 'lib64')], static: true)
#SDK_dep2 = meson.get_compiler('cpp').find_library('atutilitym', dirs: [join_paths(meson.current_source_dir(), 'lib64')], static: true)


#includes = [
#    include_directories('.') # include the main directory
#    ]

#deps = [
#	MMDevice_dep,
#	SDK_dep,
#	SDK_dep2
#	]
	
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
	'SnapShotControl.cpp',
	'SRRFAndorSDK3Camera.cpp',
]


#shared_library('AndorSDK3', sources, include_directories: includes, dependencies : deps, install: true)

