# Configuration examples

- `Emergent.ini.example` contains settings from the original test camera.
  The adapter normally creates its own file; camera sections are keyed by
  serial number. Do not overwrite an existing installation's settings file
  without reviewing its contents.
- `property-renames.json` maps older property names to current names for
  manually migrating scripts and saved hardware configurations.

Copying the adapter binary does not automatically migrate a Micro-Manager
`.cfg` file. Save a backup before applying the property-name changes.
See [the adapter guide](../docs/ADAPTER_GUIDE.md) for settings behavior.
