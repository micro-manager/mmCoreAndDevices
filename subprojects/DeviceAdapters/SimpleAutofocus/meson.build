

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FocusMonitor.cpp',
	'score.cpp',
	'SimpleAutofocus.cpp'
]


shared_library('SimpleAutofocus', sources, include_directories: includes, dependencies : deps, install: true)

