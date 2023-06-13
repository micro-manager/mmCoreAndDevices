# Device Adapter for Becker & Hickl DCC/DCU detector control cards/units

The DCC (PCIe) and DCU (USB) models use the same DLL (dcc64.dll) but a
different set of functions.

Because much of the structure is similar, we abstract the differences
(DCCDCUConfig.h) and wrap them in a (compile-time) common interface
(DCCDCUInterface.h).

The MM device objects are implemented, still parameterized by `DCCOrDCU`, in
DCCDCUDevices.h.

Finally, the devices are instantiated for DCC and DCU in DCCDCUAdapter.cpp.

---

The DCC/DCU API provided by the vendor requires that all devices ("modules") in
the same category (DCC or DCU) be initalized (and closed) by a single function
call. For this reason, these device adapters use the hub-peripheral mechanism,
where each module is a peripheral.

Also we must prevent creation of more than one hub of the same type at the same
time, because that would lead to a crash on closing.

---

Two read-only statuses ("detector overloaded" and "cooler current limit
reached") can change at any time. Because it is inconvenient for the user to
have to "Refresh" device properties to check for these conditions, we use a
background thread to poll these (see DCCDCUInterface.h).

There is a single polling thread for all DCC modules (or all DCU modules). In
order not to make assumptions about thread safety of the vendor API (no
guarantee is offered), we protect all access to the API with a mutex. This
mutex is shared between DCC and DCU.
