

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'iPadSLM.cpp',
	'Monitors.cpp',
	'OffscreenBuffer.cpp',
	'RefreshWaiter.cpp',
	'SleepBlocker.cpp',
	'SLMWindow.cpp',
	'SLMWindowClass.cpp',
	'SLMWindowThread.cpp'
]


shared_library('iPadSLM', sources, dependencies : deps, install: true)

