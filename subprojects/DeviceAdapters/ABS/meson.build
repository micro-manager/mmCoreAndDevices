

includes = [
    include_directories('.', 'include'), # include the main directory
    ]

ABSCAM_dep = meson.get_compiler('cpp').find_library('CamUsb_API64_hal', static: true, 
	dirs: join_paths(meson.current_source_dir(), 'lib_x64'))

deps = [
	boost_dep,
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


shared_library('ABS', sources, include_directories: includes, dependencies : deps, install: true)

