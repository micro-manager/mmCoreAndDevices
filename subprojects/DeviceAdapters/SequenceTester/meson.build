

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('SequenceTester', sources, dependencies : deps, install: true)

