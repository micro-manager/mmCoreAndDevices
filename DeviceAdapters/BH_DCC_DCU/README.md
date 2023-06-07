# Device Adapter for Becker & Hickl DCC/DCU detector control cards/units

The DCC (PCIe) and DCU (USB) models use the same DLL (dcc64.dll) but a
different set of functions.

Because much of the structure is similar, we abstract the differences
(DCCDCUConfig.h) and wrap them in a (compile-time) common interface
(DCCDCUInterface.h).

The MM device objects are implemented, still parameterized by `DCCOrDCU`, in
DCCDCUDevices.h.

Finally, we build separate device adapters `BH_DCC` and `BH_DCU` with a flip of
a single switch: see DCCDCUAdapter.h, DCCAdapter.cpp, and DCUAdapter.cpp.

The two adapters can be used at the same time with a single copy of dcc64.dll,
as far as I can tell from testing in simulated mode.

---

The DCC/DCU API provided by the vendor requires that all devices ("modules") in
the same category (DCC or DCU) be initalized (and closed) by a single function
call. For this reason, these device adapters use the hub-peripheral mechanism,
where each module is a peripheral.

---

Two read-only statuses ("detector overloaded" and "cooler current limit
reached") can change at any time. Because it is inconvenient for the user to
have to "Refresh" device properties to check for these conditions, we use a
background thread to poll these (see DCCDCUInterface.h).

There is a single polling thread for all DCC modules (or all DCU modules). In
order not to make assumptions about thread safety of the vendor API (no
guarantee is offered), we protect all access to the API with a mutex. However,
there is no synchronization between DCC and DCU API function calls if `BH_DCC`
and `BH_DCU` are used at the same time (let's hope this is okay; if not we'll
need to unify the two adapters under a single hub).
