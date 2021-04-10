

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Diskovery.cpp',
	'DiskoveryCommander.cpp',
	'DiskoveryListener.cpp',
	'DiskoveryModel.cpp',
	'MessageSender.cpp'
]


shared_library('Diskovery', sources, include_directories: includes, dependencies : deps, install: true)

