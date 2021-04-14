

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Arduino32bitBoards.cpp'
]


library('Arduino32bitBoards', sources, include_directories: includes, dependencies : deps, install: true)

