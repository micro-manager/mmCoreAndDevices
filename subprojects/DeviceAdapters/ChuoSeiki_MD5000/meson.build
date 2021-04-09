

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ChuoSeiki_MD_XYZ.cpp'
]


shared_library('ChuoSeiki_MD5000', sources, dependencies : deps, install: true)

