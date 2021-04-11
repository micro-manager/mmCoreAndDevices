

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XCite.cpp',
	'XCite120PC.cpp',
	'XCiteExacte.cpp'
]


shared_library('XCite120PC_Exacte', sources, include_directories: includes, dependencies : deps, install: true)

