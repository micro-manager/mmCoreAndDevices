

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ResponseDetector.cpp',
	'StringEscapes.cpp',
	'UserDefinedSerial.cpp'
]


shared_library('UserDefinedSerial', sources, include_directories: includes, dependencies : deps, install: true)

