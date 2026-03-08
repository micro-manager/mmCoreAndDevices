# MMDevice

MMDevice defines the interface between Micro-Manager device adapters and
MMCore. It also contains default implementations (in `DeviceBase.h`) for device
adapter functionality.

The device interface is a C++ interface designed to use limited features so
that binary compatibility can be maintained between compiler versions and
settings (such as MSVC Debug vs Release runtimes). This relies on the fact that
platforms maintain a stable ABI (application binary interface) for C constructs
and C++ class and vtable layout.

## Repositories

This code is available in two Git repositories:

- https://github.com/micro-manager/mmCoreAndDevices, in subdirectory
  `MMDevice`. This is the official source for MMDevice.

- https://github.com/micro-manager/mmdevice, which is a mirror of the above; we
  do not currently accept pull requests or issues on this repository.

## Building MMDevice

MMDevice is usually built as part of Micro-Manager, using its build system
(MSVC on Windows, GNU Autotools on macOS/Linux); files for that are included in
this directory (but are not intended for use in isolation).

We are in the process of modularizing the build systems of Micro-Manager's
components, and MMDevice also has cross-platform build scripts using Meson;
these can be used to build just MMDevice and run its unit tests:

```sh
git clone https://github.com/micro-manager/mmdevice.git
cd mmdevice
meson setup --vsenv builddir  # --vsenv recommended on Windows
meson compile -C builddir
meson test -C builddir
```

## Compatibility

### Device Interface Version (DIV)

The Device Interface Version (DIV) is a single integer compiled into each
device adapter (by statically linking to MMDevice). At runtime, MMCore checks
that the device adapter's DIV exactly matches its own. On mismatch, MMCore
refuses to load the adapter with an error message stating both version numbers.
The DIV is the only version number adapter authors and users need to care about.

### Cross-runtime compatibility

Given a matching DIV, adapters and MMCore built with different compilers,
compiler versions, C++ standard versions, or C++ runtimes (e.g., MSVC Debug vs
Release, libstdc++ vs libc++) are compatible. The exception is MinGW vs MSVC on
Windows (mixing is not supported due to virtual destructor incompatibility).

### Source vs binary compatibility

Source compatibility (whether adapter code compiles against a new MMDevice) is
distinct from binary compatibility (whether a compiled adapter works with a
given MMCore). A DIV bump always breaks binary compatibility but does not
necessarily break source compatibility.

Our source compatibility policy: existing device adapter source must not
silently change its semantics. If a change to MMDevice alters the contract of
an interface method, it must either (a) cause a compile error in affected
adapters, or (b) preserve the correctness of unchanged adapter source when
combined with a DIV-matched MMCore. Source-breaking changes may be made when it
is practical to safely update all affected in-tree device adapters.

### Requirements and recommendations

**DeviceBase required.** Device adapters must inherit from the DeviceBase class
templates (in `DeviceBase.h`). Deriving directly from the `MM::Device`
interface classes is not supported.

**In-tree adapters recommended.** Device adapters should be kept in-tree (in
the mmCoreAndDevices repository) so that they are automatically maintained for
source and binary compatibility. Out-of-tree adapters require careful and
regular maintenance by their authors to keep up with interface changes.

### Known portability issues

**Use of `long`.** `long` is 32-bit on Windows but 64-bit on Linux/macOS. It
is unfortunately used extensively in MMDevice, a leftover from the 32-bit days
when `long` was 32-bit on all platforms we supported. Device adapters should
assume that `long` cannot handle values outside of the range of
`std::int32_t` (and same for `unsigned long` and `std::uint32_t`).

### Recent source-incompatible changes

The following table lists source-breaking changes to MMDevice that required
updates in device adapter code. Note the reverse-chronological order.

| Date | PR | Change | Fix |
| ---- | -- | ------ | --- |
| 2026-02-26 | [#861](https://github.com/micro-manager/mmCoreAndDevices/pull/861) | Removed `GetPixelSizeUm()`, `GetComponentName()`, `PrepareSequenceAcqusition()` from `MM::Camera` interface. Removed `doProcess` parameter from `InsertImage()`. Added `UsesOnStagePositionChanged()` / `UsesOnXYStagePositionChanged()` to stage interfaces. DIV bumped to 75. | Remove `doProcess` argument from `InsertImage()` calls. Override `UsesOn[XY]StagePositionChanged()` if your stage uses position-changed callbacks. |
| 2026-02-19 | [#853](https://github.com/micro-manager/mmCoreAndDevices/pull/853) | Made `GetComponentName()` `final` | Remove override. |
| 2026-02-19 | [#852](https://github.com/micro-manager/mmCoreAndDevices/pull/852) | Made `PrepareSequenceAcqusition()` `final` | Remove override. |
| 2026-02-19 | [#845](https://github.com/micro-manager/mmCoreAndDevices/pull/845) | Moved `ImageMetadata.h` from MMDevice to MMCore | Remove `#include "ImageMetadata.h"` from adapter code. |
| 2026-02-19 | [#843](https://github.com/micro-manager/mmCoreAndDevices/pull/843) | Replaced `Metadata` with `MM::CameraImageMetadata` for camera image metadata. Removed `#include "ImageMetadata.h"` from `MMDevice.h`. | Use `MM::CameraImageMetadata` and `AddTag()` instead of `Metadata` and `PutImageTag()`. Use `md.Serialize()` (returns `const char*`) instead of `md.Serialize().c_str()`. |
| 2026-02-09 | [#832](https://github.com/micro-manager/mmCoreAndDevices/pull/832) | Removed `GetTagKeys()` and `GetTagValue()` from `CCameraBase` | Stop calling these methods. |
| 2026-02-06 | [#827](https://github.com/micro-manager/mmCoreAndDevices/pull/827) | Removed `ImgBuffer::SetMetadata()` and `GetMetadata()` | Stop calling these methods. |
| 2025-10-30 | [#771](https://github.com/micro-manager/mmCoreAndDevices/pull/771) | Deleted `_t()` timing macro | Remove uses of `_t()`. |
| 2025-09-26 | [#752](https://github.com/micro-manager/mmCoreAndDevices/pull/752) | Made `GetPixelSizeUm()` `final` in camera base classes | Remove override of `GetPixelSizeUm()`. |
| 2025-09-25 | [#753](https://github.com/micro-manager/mmCoreAndDevices/pull/753) | Removed `using` declarations for `CreateProperty`, `SetAllowedValues`, `SetProperty` from `CCameraBase` (reverted from public to protected) | No change needed if calling from within the class hierarchy (normal case). |
| 2025-09-25 | [#590](https://github.com/micro-manager/mmCoreAndDevices/pull/590) | Split `CCameraBase` into `CCameraBase` (new; cameras correctly implementing sequence acquisition) and `CLegacyCameraBase` (cameras relying on deprecated snap-thread behavior, or parts of it) | Cameras using the old snap-thread pattern: change base class to `CLegacyCameraBase<T>`. New `CCameraBase` also requires implementing `Busy()`. |
| 2025-08-15 | [#710](https://github.com/micro-manager/mmCoreAndDevices/pull/710), [#715](https://github.com/micro-manager/mmCoreAndDevices/pull/715), [#717](https://github.com/micro-manager/mmCoreAndDevices/pull/717) | Removed deprecated Core callback functions (`GetXYPosition`, `ClearImageBuffer`, `InsertImage(ImgBuffer)`, `InsertMultiChannel`, `GetImage`, `GetImageDimensions`, `SetFocusPosition`, `MoveFocus`, `SetXYPosition`, `MoveXYStage`, `SetExposure`, `GetExposure`, `SetConfig`, `GetCurrentConfig`, `GetChannelConfig`, `GetImageProcessor`, `GetAutoFocus`, `GetStateDevice`, `NextPostedError`, `PostError`, `ClearPostedErrors`). Removed `HDEVMODULE` type, `GetModuleHandle()`, `SetModuleHandle()`. DIV bumped to 74. | Remove calls to these functions. For `ClearImageBuffer`: just stop acquisition on any `InsertImage()` error. |
| 2024-01-24 | [#435](https://github.com/micro-manager/mmCoreAndDevices/pull/435) | Replaced `MODULE_EXPORTS` preprocessor macro with `MMDEVICE_CLIENT_BUILD` (with inverted logic; adapters no longer need to define any macro). | Remove `MODULE_EXPORTS` from preprocessor definitions. |
| 2024-01-22 | [#433](https://github.com/micro-manager/mmCoreAndDevices/pull/433) | Removed `gettimeofday()` shim, `DELTA_EPOCH_IN_MICROSECS` macro, and `struct timezone` from `DeviceUtils.h`. Removed transitive `<windows.h>` include. | Remove uses of `gettimeofday()`. Add explicit `<windows.h>` include if needed for Windows-specific types. |
| 2023-08-26 | [#367](https://github.com/micro-manager/mmCoreAndDevices/pull/367) | Deleted `FixSnprintf.h` and removed its `#include` from `MMDevice.h`. | Remove `#include "FixSnprintf.h"` from adapter code. |
| 2022-10-27 | [#277](https://github.com/micro-manager/mmCoreAndDevices/pull/277), [#278](https://github.com/micro-manager/mmCoreAndDevices/pull/278), [#279](https://github.com/micro-manager/mmCoreAndDevices/pull/279), [#280](https://github.com/micro-manager/mmCoreAndDevices/pull/280), [#281](https://github.com/micro-manager/mmCoreAndDevices/pull/281) | `MM::MMTime` overhaul: removed `MMTime(std::string)` constructor, made constructors `explicit`, made `sec_`/`uSec_` data members private, replaced `serialize()` with `toString()`, changed `fromUs()` parameter from `double` to `long long`, removed `MM::g_Keyword_Metadata_StartTime` constant. | Use `MM::MMTime{}` or `MM::MMTime(0)` instead of implicit conversions. Access time via `getMicroseconds()`, `fromUs()`, `fromMs()`. Use `toString()` instead of `serialize()`. Remove references to `g_Keyword_Metadata_StartTime`. |
| 2021-02-14 | [SVN r17423](https://github.com/micro-manager/mmCoreAndDevices/commit/8687ddb51) | Removed unused `AcqBefore()`, `AcqAfter()`, `AcqBeforeFrame()`, `AcqAfterFrame()`, `AcqBeforeStack()`, `AcqAfterStack()` from `MM::Device` and `CDeviceBase`. DIV bumped to 70. | Remove overrides of these methods. |
| 2017-01-30 | [SVN r16281](https://github.com/micro-manager/mmCoreAndDevices/commit/e348a6f1a) | Removed `FrameBuffer` class from `ImgBuffer.h`. | Remove uses of `FrameBuffer`. |
| 2016-01-06 | [SVN r15892](https://github.com/micro-manager/mmCoreAndDevices/commit/f53fa08d2) | `Property`, `StringProperty`, `FloatProperty`, `IntegerProperty` constructors now require a `name` parameter. | Use `CreateProperty()` and family instead of direct constructor call. |

> This table covers changes since early 2016. History back to 2014 has been
> reviewed; the next source-breaking changes before this period were in
> mid-2014 (DIV 55–61 era).

## Device Interface Version policy

_This section is for MMDevice/MMCore maintainers._

These are general guidelines but may not be perfect for every situation. They
do not fully replace the need for careful consideration and review of each
proposed change. Also, don't forget that compatibility is not the only concern
in good interface design.

### Where versions are defined

The Device Interface Version (DIV) is defined as `DEVICE_INTERFACE_VERSION` in
`MMDevice.h`. The Module Interface Version (MIV) is defined as
`MODULE_INTERFACE_VERSION` in `ModuleInterface.h`. Both are compiled into each
device adapter (via `ModuleInterface.cpp`) and checked by MMCore at load time
(exact match required for both).

### Binary compatibility mechanism

The interface classes in `MMDevice.h` define pure virtual methods. MMCore calls
device methods exclusively through virtual dispatch (vtable). Similarly,
`MM::Core` is a virtual callback interface implemented by MMCore and called by
adapters. Binary compatibility means the vtable layout, method signatures, and
data passed across the boundary must be identical on both sides.

### Building against matching MMDevice

Each device adapter (and MMCore) must individually be built against the
matching MMDevice version from the same mmCoreAndDevices commit. There is no
support for building against an older or newer MMDevice. However, adapters and
MMCore built from different commits can be freely mixed as long as their DIV
matches.

### Constraints on interface classes

The following constraints are enforced to maintain ABI portability:

- Method parameters and return values must be POD types or pointers; no C++
  types (`std::string`, `std::vector`, etc.) may cross the boundary.
  - Note: `MM::MMTime` and `MM::Core::GetCurrentMMTime()` violate this rule.
    `MM::MMTime` is slated for removal from the binary interface.
- No cross-interface memory allocation/deallocation. Memory allocated on one
  side must be freed on the same side.
- Single inheritance only (no multiple inheritance, which, while theoretically
  might work, is likely to paint us into a corner regarding future possibilities
  such as MinGW compatibility or a pure C device interface).
- No exceptions may propagate across the adapter-Core boundary. Adapters and
  MMCore are free to use exceptions internally.
- No RTTI (`dynamic_cast`, `typeid`) in the interface classes. Adapters may
  use RTTI internally.
- No covariant return types in virtual methods. Some compilers implement
  covariant returns via thunks that alter vtable layout, breaking
  cross-compiler compatibility.

### Changes that require a DIV bump

- Adding, removing, or reordering virtual methods in any interface class,
  including `MM::Core` (the callback interface).
- Changing the signature of any virtual method (parameters, return type, const
  qualification).
- Adding or removing members in any struct whose instances cross the
  adapter-Core boundary (such structs do not currently exist but may be used in
  the future).
- Adding a new device type (a new class inheriting from `MM::Device`), which is
  a compound change touching `MMDevice.h`, `MMDeviceConstants.h` (`DeviceType`
  enum), `DeviceBase.h`, and `ModuleInterface.h`/`.cpp`.
- Changing the camera image metadata serialization format so that
  `CameraImageMetadata::Serialize()` produces output not accepted by all
  DIV-matching versions of MMCore. (`CameraImageMetadata` is the structured
  metadata that cameras pass alongside image data via `InsertImage()`.) Other
  changes to the metadata classes may not require a DIV bump, but must be
  verified by reviewing all changes to both MMCore's deserialization and
  `CameraImageMetadata::Serialize()` since the last DIV bump.
- Any change to `ModuleInterface.h` exported functions (Module Interface Version
  bump) — the DIV must always be bumped when the MIV is bumped (not vice
  versa), so that the DIV alone suffices as the user-facing compatibility
  version. (The pymmcore ecosystem uses the DIV as its compatibility version
  number: pymmcore versions embed the DIV, and pymmcore-plus uses it to install
  matching components.)
- Adding or removing `noexcept` on a virtual method can change vtable layout
  in some ABIs. Also, `noexcept` should not be added to interface methods
  until C++14 support is dropped, because C++17 made `noexcept` part of the
  type system.
- Adding or removing enums or constants in `MMDeviceConstants.h`, if they
  cross the adapter-Core boundary in a way that would cause incompatibility
  (this is case-by-case).
- Any change that would cause an adapter built after the change to be
  incompatible with a Core built before the change (at the same DIV), or vice
  versa — including behavioral changes in `DeviceBase.h` that affect the
  adapter-Core interaction.

### Changes that do NOT require a DIV bump

- Changes to MMCore internals that don't affect the device interface.
- Changes to `DeviceBase.h` that only affect (at most) source compatibility
  (e.g., renaming a helper, changing a default implementation in a way that is a
  uniform improvement and doesn't break adapter-Core interaction).
- Adding new non-virtual utility functions or classes that don't cross the
  boundary.

### Source compatibility notes

Adding overloads to existing methods must be done carefully so that implicit
conversions don't silently redirect existing calls to a different overload,
changing semantics. Similarly, removal of overloads must ensure that
previously-valid calls produce a compile error rather than silently binding to a
remaining overload via implicit conversion (unless the remaining overload is
semantically equivalent).

Default argument values are compiled into the caller. Changing a default value
silently changes behavior for adapters built against the old default, and can
cause a mismatch between what MMCore expects and the default a device adapter
was compiled with.

### DeviceBase.h default implementation technique

When adding a new virtual method to an interface class (which requires a DIV
bump), a default implementation in `DeviceBase.h` can shield adapter source
from the change — adapters that don't override the new method will compile and
work correctly with the default.

### MMDeviceConstants.h and the public API

`MMDeviceConstants.h` is part of the MMCore public API, and transitively the
MMCoreJ (Java) and pymmcore (Python) public APIs. Removal of definitions must
be coordinated with all those projects. Changing an existing definition (as
opposed to removal) is almost never a good idea, even if it contains a typo.
