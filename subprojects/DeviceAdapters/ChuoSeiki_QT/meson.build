

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ChuoSeiki_QT_XYZ.cpp'
]


shared_library('ChuoSeiki_QT', sources, dependencies : deps, install: true)

