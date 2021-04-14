

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'InterDevice.cpp',
	'LoggedSetting.cpp',
	'SequenceTester.cpp',
	'SettingLogger.cpp',
	'TextImage.cpp',
	'TriggerInput.cpp'
]


library('SequenceTester', sources, include_directories: includes, dependencies : deps, install: true)

