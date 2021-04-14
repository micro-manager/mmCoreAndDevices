

includes = [
    include_directories('.'), # include the main directory
    ]

ABSCAM_dep = dependency('ABS_CamUsb', fallback: ['3rdPartyPrivate'])

deps = [
	MMDevice_dep,
	ABSCAM_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ABSCamera.cpp',
	'abscommontools.cpp',
	'absdelayloaddll.cpp',
	'AbsImgBuffer.cpp',
	'AutoTimeMeasure.cpp',
	'CcmFileStd.cpp',
	'dllmain.cpp',
	'ImageHeader.cpp',
	'MemoryIniFile.cpp',
	'TimeSpanPC.cpp'
]


library('ABS', sources, include_directories: includes, dependencies : deps, install: true)

