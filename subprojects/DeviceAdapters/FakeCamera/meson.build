

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'error_code.cpp',
	'FakeCamera.cpp',
	'module.cpp'
]


shared_library('FakeCamera', sources, include_directories: includes, dependencies : deps, install: true)

