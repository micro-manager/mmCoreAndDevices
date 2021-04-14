

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoboltOfficial.cpp',
	'DeviceProperty.cpp',
	'Dpl06Laser.cpp',
	'EnumerationProperty.cpp',
	'ImmutableEnumerationProperty.cpp',
	'Laser.cpp',
	'LaserFactory.cpp',
	'LaserShutterProperty.cpp',
	'LaserStateProperty.cpp',
	'Mld06Laser.cpp',
	'MutableDeviceProperty.cpp',
	'NoShutterCommandLegacyFix.cpp',
	'NumericProperty.cpp',
	'Property.cpp',
	'SkyraLaser.cpp',
	'StaticStringProperty.cpp'
]


library('CoboltOfficial', sources, include_directories: includes, dependencies : deps, install: true)

