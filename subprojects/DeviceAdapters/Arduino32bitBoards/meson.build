

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Arduino32bitBoards.cpp'
]


shared_library('Arduino32bitBoards', sources, dependencies : deps, install: true)

