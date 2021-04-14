

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ResponseDetector.cpp',
	'StringEscapes.cpp',
	'UserDefinedSerial.cpp'
]


library('UserDefinedSerial', sources, include_directories: includes, dependencies : deps, install: true)

