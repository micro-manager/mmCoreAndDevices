#include "ModuleInterface.h"
#include "DC40.h"

#include "include/TLDC.h"

const char* g_DeviceDC40Name = "Thorlabs DC40";

MODULE_API void InitializeModuleData()
{
    // Find and connect to the first available device
    ViUInt32 numDevices;
    int error = TLDC_findRsrc(VI_NULL, &numDevices);
    if (error) return;

    for (ViUInt32 i = 0; i < numDevices; i++) {
        ViChar resourceName[TLDC_BUFFER_SIZE];
        ViChar serNr[TLDC_BUFFER_SIZE];
        ViPBoolean available = false;
        error = TLDC_getRsrcInfo(0, i, resourceName, serNr, VI_NULL, available);
        if (error) return;  // TODO: LOG!!!
        RegisterDevice(serNr, MM::GenericDevice, "DC40 4.0 A LED Driver");
    }
}

/*---------------------------------------------------------------------------
 Creates and returns a device specified by the device name.
---------------------------------------------------------------------------*/
MODULE_API MM::Device* CreateDevice(const char* serialNr)
{
    // no name, no device
    if (serialNr == 0)	return 0;

    return new DC40(serialNr);
}

/*---------------------------------------------------------------------------
 Deletes a device pointed by pDevice.
---------------------------------------------------------------------------*/
MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
