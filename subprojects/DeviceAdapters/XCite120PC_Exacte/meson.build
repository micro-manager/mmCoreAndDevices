

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XCite.cpp',
	'XCite120PC.cpp',
	'XCiteExacte.cpp'
]


shared_library('XCite120PC_Exacte', sources, dependencies : deps, install: true)

