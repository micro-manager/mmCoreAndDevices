// #pragma warning(push)
// #pragma warning(disable : 4482)
// #pragma warning(disable : 4251) // Note: need to have a C++ interface, i.e., compiler versions need to match!

#include "AravisCamera.h"
#include "ModuleInterface.h"
#include <vector>
#include <string>
#include <algorithm>


/*
 * Module functions.
 */
MODULE_API void InitializeModuleData()
{
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
  return new AravisCamera(deviceName);
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
}


/*
 * Camera class and methods.
 */

AravisCamera::AravisCamera(const char *name) : CCameraBase<AravisCamera>()
{
}

AravisCamera::~AravisCamera()
{
}


/*
 * Acquistion thread class and methods.
 */
AravisAcquisitionThread::AravisAcquisitionThread(AravisCamera * aCam)
{
}

AravisAcquisitionThread::~AravisAcquisitionThread()
{
}

// #pragma warning(pop)
