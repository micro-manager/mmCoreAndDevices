doxygen = find_program(
    'doxygen',
    required: get_option('docs'),
    disabler: true,
)

doxygen_werror = get_option('docs_werror') ? 'FAIL_ON_WARNINGS' : 'NO'

project_root = meson.current_source_dir() / '..'

doxyfile = configure_file(
    configuration: {
        'DOXYGEN_INPUT': project_root,
        'DOXYGEN_OUTPUT': meson.current_build_dir() / 'apidoc',
        'DOXYGEN_STRIP_PATH': project_root,
        'DOXYGEN_EXCLUDE': project_root / 'subprojects',
        'DOXYGEN_WERROR': doxygen_werror,
    },
    input: 'Doxyfile.in',
    output: 'Doxyfile',
)

doc_install_dir = get_option('datadir') / 'doc' / meson.project_name()

custom_target(
    'docs',
    command: [doxygen, '-q', doxyfile],
    output: ['apidoc'],
    install: true,
    install_dir: doc_install_dir,
)
