

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'BiostepEagleIce.cpp'
]


shared_library('BiostepEagleIce', sources, dependencies : deps, install: true)

