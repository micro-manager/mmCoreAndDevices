

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AcqConfig.cpp',
	'AcqThread.cpp',
	'Event.cpp',
	'NotificationEntry.cpp',
	'NotificationThread.cpp',
	'PollingThread.cpp',
	'PpParam.cpp',
	'PVCAMAdapter.cpp',
	'PVCAMParam.cpp',
	'PVCAMUniversal.cpp',
	'PvCircularBuffer.cpp',
	'PvDebayer.cpp',
	'PvFrameInfo.cpp',
	'PvRoiCollection.cpp'
]


shared_library('PVCAM', sources, include_directories: includes, dependencies : deps, install: true)

