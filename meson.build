project('mmCoreAndDevices', 'cpp', version : '0.0.1',
	default_options: [
		'prefix=@0@'.format(join_paths(meson.project_source_root(), 'stage'))
	]
)


subproject('MMDevice')
subproject('MMCore')

if get_option('BUILD_MMCOREJ')
	subproject('MMCoreJ_wrap')
endif

if get_option('BUILD_DEVICE_ADAPTERS')
	subproject('DeviceAdapters')
endif

if get_option('BUILD_MMCOREPY')
	subproject('MMCorePy_wrap')
endif