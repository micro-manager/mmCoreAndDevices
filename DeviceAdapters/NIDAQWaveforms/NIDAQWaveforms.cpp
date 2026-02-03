///////////////////////////////////////////////////////////////////////////////
// FILE:          NIDAQWaveforms.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generates analog waveforms on a NI-DAQ.
//                
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2026
//

#include "NIDAQWaveforms.h"
#include "MockDAQAdapter.h"
#include "NIDAQmxAdapter.h"
#include "ModuleInterface.h"

using namespace std;

const char* g_DeviceName = "NIDAQ Waveforms";
const char* g_DeviceDescription = "Generates analog waveforms on a NI-DAQ device";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::GenericDevice, "NIDAQ Waveforms");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      // create the test device
      return new NIDAQWaveforms();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// NIDAQWaveforms implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* NIDAQWaveforms constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
NIDAQWaveforms::NIDAQWaveforms() :
	initialized_ (false),
	deviceName_("")
{
	InitializeDefaultErrorMessages();

	// Create DAQ adapter for device discovery
	// Toggle between Mock and NIDAQmx by commenting/uncommenting:
	daq_ = std::make_unique<MockDAQAdapter>();
	// daq_ = std::make_unique<NIDAQmxAdapter>();

	// Pre-init property: Device
	std::vector<std::string> devices = daq_->getDeviceNames();
	CPropertyAction* pAct = new CPropertyAction(this, &NIDAQWaveforms::OnDevice);
	std::string defaultDevice = devices.empty() ? "" : devices[0];
	deviceName_ = defaultDevice;
	CreateStringProperty("Device", defaultDevice.c_str(), false, pAct, true);
	for (const auto& dev : devices)
		AddAllowedValue("Device", dev.c_str());
}

/**
* NIDAQWaveforms destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
NIDAQWaveforms::~NIDAQWaveforms()
{
   if (initialized_)
      Shutdown();
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void NIDAQWaveforms::GetName(char* name) const
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
int NIDAQWaveforms::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	// set read-only properties
	// ------------------------
	int nRet = CreateStringProperty(MM::g_Keyword_Name, g_DeviceName, true);
	if (DEVICE_OK != nRet)
		return nRet;

	nRet = CreateStringProperty(
		MM::g_Keyword_Description,
		g_DeviceDescription,
		true
	);
	if (DEVICE_OK != nRet)
		return nRet;

	// Discover available AO channels for the selected device
	availableChannels_ = daq_->getAnalogOutputChannels(deviceName_);

	// Create enable/disable property for each channel
	for (const auto& channel : availableChannels_)
	{
		std::string propName = channel + " Enabled";
		CPropertyAction* pAct = new CPropertyAction(this, &NIDAQWaveforms::OnChannelEnabled);
		nRet = CreateStringProperty(propName.c_str(), "No", false, pAct);
		if (DEVICE_OK != nRet)
			return nRet;
		AddAllowedValue(propName.c_str(), "No");
		AddAllowedValue(propName.c_str(), "Yes");
	}

	// synchronize all properties
	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;
	return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int NIDAQWaveforms::Shutdown()
{
   daq_.reset();
   initialized_ = false;
   return DEVICE_OK;
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

/////////////////////////////////////////////
// Action handlers
/////////////////////////////////////////////

int NIDAQWaveforms::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(deviceName_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(deviceName_);
	}
	return DEVICE_OK;
}

int NIDAQWaveforms::OnChannelEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	
	if (eAct == MM::AfterSet)
	{
		// Rebuild the list of enabled channels from properties
		enabledChannels_.clear();
		for (const auto& channel : availableChannels_)
		{
			std::string propName = channel + " Enabled";
			char value[MM::MaxStrLength];
			GetProperty(propName.c_str(), value);
			if (std::string(value) == "Yes")
				enabledChannels_.push_back(channel);
		}
	}
	return DEVICE_OK;
}
