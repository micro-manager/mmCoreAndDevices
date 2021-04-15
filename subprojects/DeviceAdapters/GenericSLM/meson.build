

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	dependency('boost'),
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'GenericSLM.cpp',
	'Monitors.cpp',
	'OffscreenBuffer.cpp',
	'RefreshWaiter.cpp',
	'SleepBlocker.cpp',
	'SLMWindow.cpp',
	'SLMWindowClass.cpp',
	'SLMWindowThread.cpp'
]


library('GenericSLM', sources, include_directories: includes, dependencies : deps, install: true)

