///////////////////////////////////////////////////////////////////////////////
// FILE:          ArduinoShutter.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   A basic Arduino-based shutter
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

#include "ArduinoShutter.h"
#include "ModuleInterface.h"

using namespace std;

const char* g_DeviceName = "ArduinoShutter";
const char* g_DeviceDescription = "A basic Arduino-based shutter";

const char* g_OpenCmd = "open";
const char* g_CloseCmd = "close";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
   // Note the device type is specified here!
   RegisterDevice(g_DeviceName, MM::ShutterDevice, "Arduino Shutter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      // create the test device
      return new ArduinoShutter();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// ArduinoShutter implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* ArduinoShutter constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
ArduinoShutter::ArduinoShutter() :
	// Parameter values before hardware synchronization
	initialized_ (false),
	open_ (false),
	msg_(""),
	response_("")
{
	// call the base class method to set-up default error codes/messages
	InitializeDefaultErrorMessages();

	GeneratePreInitProperties();

	EnableDelay();
}

/**
* ArduinoShutter destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
ArduinoShutter::~ArduinoShutter()
{
   if (initialized_)
      Shutdown();
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void ArduinoShutter::GetName(char* name) const
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
int ArduinoShutter::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	int ret = GenerateReadOnlyProperties();
	if (DEVICE_OK != ret) return ret;

	ret = GenerateControlledProperties();
	if (DEVICE_OK != ret) return ret;

    // synchronize all properties
    // --------------------------
    ret = UpdateStatus();
    if (ret != DEVICE_OK) return ret;

    initialized_ = true;
    return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int ArduinoShutter::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

/////////////////////////////////////////////
// MM:Shutter API
// These are called by Micro-Manager and do 
// not need to be called directly by our
// code.
/////////////////////////////////////////////
int ArduinoShutter::SetOpen(bool open)
{
	long pos;
	if (open)
		pos = 1;
	else
		pos = 0;
	return SetProperty(MM::g_Keyword_State, CDeviceUtils::ConvertToString(pos));
}

int ArduinoShutter::GetOpen(bool& open)
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(MM::g_Keyword_State, buf);
	if (ret != DEVICE_OK) return ret;
	
	long pos = atol(buf);
	pos == 1 ? open = true : open = false;

	return DEVICE_OK;
}
int ArduinoShutter::Fire(double /*deltaT*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////
int ArduinoShutter::GeneratePreInitProperties() {
	CPropertyAction* pAct = new CPropertyAction(this, &ArduinoShutter::OnPort);
	return CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

int ArduinoShutter::GenerateReadOnlyProperties() {
	// Name
	int ret = CreateStringProperty(MM::g_Keyword_Name, g_DeviceName, true);
	if (DEVICE_OK != ret) return ret;

	// Description
	ret = CreateStringProperty(MM::g_Keyword_Description, g_DeviceDescription, true);
	if (DEVICE_OK != ret) return ret;

	CPropertyAction* pAct = new CPropertyAction(this, &ArduinoShutter::OnResponseChange);
	ret = CreateStringProperty("Response", response_.c_str(), true, pAct);
	if (DEVICE_OK != ret) return ret;

	return DEVICE_OK;
}

int ArduinoShutter::GenerateControlledProperties() {
	CPropertyAction* pAct = new CPropertyAction(this, &ArduinoShutter::OnStateChange);
	int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
	if (DEVICE_OK != ret) return ret;

	SetPropertyLimits(MM::g_Keyword_State, 0, 1);

	return DEVICE_OK;

}

/////////////////////////////////////////////
// Action handlers
/////////////////////////////////////////////
int ArduinoShutter::OnStateChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Simply return the current state
		pProp->Set(open_ ? (long)1 : (long)0);
	}
	else if (eAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);
		bool mmWantsOpen = (pos > 0);

		// Only send the command if the property changed
		if (open_ != mmWantsOpen) {
			int ret;
			if (mmWantsOpen) {
				ret = this->QueryDevice(g_OpenCmd);
				if (ret != DEVICE_OK) return ret;
			}
			else {
				ret = this->QueryDevice(g_CloseCmd);
				if (ret != DEVICE_OK) return ret;
			}

			// Update the state only after successful communication
			open_ = mmWantsOpen;
		}
	}

	return DEVICE_OK;
}

int ArduinoShutter::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int ArduinoShutter::OnResponseChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(response_.c_str());
	}

	return DEVICE_OK;
}


/////////////////////////////////////////////
// Serial communications
/////////////////////////////////////////////
int ArduinoShutter::PurgeBuffer()
{
	int ret = PurgeComPort(port_.c_str());
	if (ret != DEVICE_OK)
		return DEVICE_SERIAL_COMMAND_FAILED;

	return DEVICE_OK;
}

/**
 * Sends a command to the device and receives the result.
 */
int ArduinoShutter::QueryDevice(std::string msg)
{

	int ret = PurgeBuffer();
	if (ret != DEVICE_OK) return ret;
	
	ret = SendMsg(msg);
	if (ret != DEVICE_OK) return ret;

	ret = ReceiveMsg();
	if (ret != DEVICE_OK) return ret;

	return DEVICE_OK;
}

int ArduinoShutter::ReceiveMsg()
{
	std::string valid = "";
	int ret;

	// Get the data returned by the device.
	ret = GetSerialAnswer(port_.c_str(), ANS_TERM_.c_str(), response_);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int ArduinoShutter::SendMsg(std::string msg)
{
	int ret = SendSerialCommand(port_.c_str(), msg.c_str(), CMD_TERM_.c_str());
	if (ret != DEVICE_OK)
		return DEVICE_SERIAL_COMMAND_FAILED;

	return DEVICE_OK;
}