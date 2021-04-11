

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
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


shared_library('GenericSLM', sources, include_directories: includes, dependencies : deps, install: true)

