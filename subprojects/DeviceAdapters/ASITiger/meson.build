

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ASIClocked.cpp',
	'ASICRISP.cpp',
	'ASIDac.cpp',
	'ASIFWheel.cpp',
	'ASIHub.cpp',
	'ASILED.cpp',
	'ASILens.cpp',
	'ASIPiezo.cpp',
	'ASIPLogic.cpp',
	'ASIPmt.cpp',
	'ASIScanner.cpp',
	'ASITiger.cpp',
	'ASITigerComm.cpp',
	'ASIXYStage.cpp',
	'ASIZStage.cpp'
]


shared_library('ASITiger', sources, dependencies : deps, install: true)

