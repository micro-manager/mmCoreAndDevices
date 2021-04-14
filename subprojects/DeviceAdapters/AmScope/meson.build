
SDK_dep = dependency('ToupCamSDK', fallback: '3rdPartyPrivate')

includes = [
    include_directories('.'), # include the main directory
    ]

deps = [
	MMDevice_dep,
	SDK_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AmScope.cpp',
	'SequenceThread.cpp'
]


library('AmScope', sources, include_directories: includes, dependencies : deps, install: true)

