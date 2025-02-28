///////////////////////////////////////////////////////////////////////////////
// FILE:          SkeletonSerial.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Skeleton adapter for a serial port device
//                
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2025
//

#include "SkeletonSerial.h"
#include "ModuleInterface.h"

using namespace std;

const char* g_DeviceName = "SkeletonSerial";
const char* g_DeviceDescription = "Skeleton adapter for a serial port device";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::GenericDevice, "Skeleton Device Adapter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      // create the test device
      return new SkeletonSerial();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// SkeletonSerial implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* SkeletonSerial constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
SkeletonSerial::SkeletonSerial() :
	// Parameter values before hardware synchronization
	initialized_ (false)
{
	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();

	// Setup the port
	CPropertyAction* pAct = new CPropertyAction(this, &SkeletonSerial::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

/**
* SkeletonSerial destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
SkeletonSerial::~SkeletonSerial()
{
   if (initialized_)
      Shutdown();
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void SkeletonSerial::GetName(char* name) const
{
   // We just return the name we use for referring to this
   // device adapter.
   CDeviceUtils::CopyLimitedString(name, g_DeviceName);
}

/**
* Intializes the hardware.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well.
* Required by the MM::Device API.
*/
int SkeletonSerial::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	// set read-only properties
	// ------------------------
	// Name
	int nRet = CreateStringProperty(MM::g_Keyword_Name, g_DeviceName, true);
	if (DEVICE_OK != nRet)
		return nRet;

	// Description
	nRet = CreateStringProperty(
		MM::g_Keyword_Description,
		g_DeviceDescription,
		true
	);
	if (DEVICE_OK != nRet)
		return nRet;

    // synchronize all properties
    // --------------------------
    int ret = UpdateStatus();
    if (ret != DEVICE_OK)
       return ret;

    initialized_ = true;
    return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int SkeletonSerial::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

/////////////////////////////////////////////
// Action handlers
/////////////////////////////////////////////
int SkeletonSerial::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}

		pProp->Get(port_);
	}

	return DEVICE_OK;
}


/////////////////////////////////////////////
// Serial communications
/////////////////////////////////////////////
const std::string SkeletonSerial::GetLastMsg()
{
	return buffer_;
}

int SkeletonSerial::PurgeBuffer()
{
	int ret = PurgeComPort(port_.c_str());
	if (ret != DEVICE_OK)
		return DEVICE_SERIAL_COMMAND_FAILED;

	return DEVICE_OK;
}

/**
 * Sends a command to the device and receives the result.
 */
std::string SkeletonSerial::QueryDevice(std::string msg)
{
	std::string result;

	PurgeBuffer();
	SendMsg(msg);
	ReceiveMsg();
	result = GetLastMsg();

	return result;
}

int SkeletonSerial::ReceiveMsg()
{
	std::string valid = "";
	int ret;

	// Get the data returned by the device.
	ret = GetSerialAnswer(port_.c_str(), ANS_TERM_.c_str(), buffer_);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int SkeletonSerial::SendMsg(std::string msg)
{
	int ret = SendSerialCommand(port_.c_str(), msg.c_str(), CMD_TERM_.c_str());
	if (ret != DEVICE_OK)
		return DEVICE_SERIAL_COMMAND_FAILED;

	return DEVICE_OK;
}