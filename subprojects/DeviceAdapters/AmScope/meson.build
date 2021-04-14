
SDK_dir = join_paths(MM_3RDPARTY_PRIVATE, 'Touptek', 'toupcamsdk', '20160920')
SDK_dep = meson.get_compiler('cpp').find_library('toupcam', dirs: [join_paths(SDK_dir,  'win', 'x64')], static: true)

includes = [
    include_directories('.'), # include the main directory
	include_directories(join_paths(SDK_dir, 'inc'))
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

