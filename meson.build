project('mmCoreAndDevices', 'cpp', version : '0.0.1',
	default_options: [
		'prefix=@0@'.format(join_paths(meson.project_source_root(), 'stage'))
	]
)


subproject('MMDevice')
subproject('MMCore')
subproject('MMCoreJ_wrap')
subproject('DeviceAdapters')
