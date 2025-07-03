///////////////////////////////////////////////////////////////////////////////
// FILE:          Illuminator.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Zaber's X-LCA series LED drivers.
//
// AUTHOR:        Soleil Lapierre (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies, 2019

// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifdef WIN32
#pragma warning(disable: 4355)
#endif

#include "Illuminator.h"

const char* g_IlluminatorName = "Illuminator";
const char* g_IlluminatorDescription = "Zaber Illuminator";

using namespace std;

Illuminator::Illuminator()
: ZaberBase(this)
, deviceAddress_(1)
, numLamps_(0)
, lampExists_(NULL)
, canUseDeviceLampOnCommand_(true)
, currentFlux_(NULL)
, maxFlux_(NULL)
, lampIsOn_(NULL)
, isOpen_(false)
{
	LogMessage("Illuminator::Illuminator\n", true);

	InitializeDefaultErrorMessages();
	ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });

	// Pre-initialization properties
	CreateProperty(MM::g_Keyword_Name, g_IlluminatorName, MM::String, true);

	CreateProperty(MM::g_Keyword_Description, "Zaber illuminator device adapter", MM::String, true);

	CPropertyAction* pAct = new CPropertyAction(this, &Illuminator::PortGetSet);
	CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

	pAct = new CPropertyAction (this, &Illuminator::DeviceAddressGetSet);
	CreateIntegerProperty("Controller Device Number", deviceAddress_, false, pAct, true);
	SetPropertyLimits("Controller Device Number", 1, 99);
}


Illuminator::~Illuminator()
{
	LogMessage("Illuminator::~Illuminator\n", true);
	Shutdown();
}


///////////////////////////////////////////////////////////////////////////////
// Device API methods
///////////////////////////////////////////////////////////////////////////////

void Illuminator::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_IlluminatorName);
}


int Illuminator::Initialize()
{
	if (initialized_)
	{
		return DEVICE_OK;
	}

	core_ = GetCoreCallback();

	LogMessage("Illuminator::Initialize\n", true);

	// Check the firmware version, ignoring any warning flags.
	double version;
	auto ret = GetFirmwareVersion(deviceAddress_, version);
	if (ret != DEVICE_OK)
	{
		LogMessage("Firmware version read failed.\n", true);
		return ret;
	}
	else if (version < 7.14)
	{
		LogMessage("Firmware version is too old.\n", true);
		return ERR_FIRMWARE_UNSUPPORTED;
	}

	// Activate any recently changed peripherals.
	ret = ActivatePeripheralsIfNeeded(deviceAddress_);
	if (ret != DEVICE_OK)
	{
		LogMessage("Peripheral activation check failed.\n", true);
		return ret;
	}

	// Detect the number of LEDs.
	long data = 0;
	ret = GetSetting(deviceAddress_, 0, "system.axiscount", data);
	if (ret != DEVICE_OK)
	{
		LogMessage("Failed to detect the number of lamps in the illuminator.\n", true);
		return ret;
	}
	else if ((data < 1) || (data > 256)) // Arbitrary upper limit.
	{
		LogMessage("Number of lamps is out of range.\n", true);
		return DEVICE_NOT_SUPPORTED;
	}

	numLamps_ = (int)data;
	currentFlux_ = new double[numLamps_];
	maxFlux_ = new double[numLamps_];
	lampExists_ = new bool[numLamps_];
	lampIsOn_ = new bool[numLamps_];

	// Read the properties of each lamp.
	double analogData = 0.0;
	CPropertyActionEx* action;
	char nameBuf[256];
	char valueBuf[128];
	canUseDeviceLampOnCommand_ = true;

	RefreshLampStatus();

	snprintf(valueBuf, sizeof(valueBuf), "%d", numLamps_);
	CreateProperty("Number of lamps", valueBuf, MM::Integer, true);

	for (int i = 0; i < numLamps_; ++i)
	{
		// These properties (especially the writable ones) should always be
		// created for all axes even if the lamp doesn't exist, because
		// otherwise if someone unplugs a lamp they will get errors when
		// loading their config, and will not be able to edit their presets.


		snprintf(nameBuf, sizeof(nameBuf) - 1, "Lamp %d connected", i + 1);
		CreateProperty(nameBuf, lampExists_[i] ? "1" : "0", MM::Integer, true);

		data = 0;
		if (lampExists_[i])
		{
			ret = GetSetting(deviceAddress_, i + 1, "lamp.wavelength.peak", data);
			if (ret != DEVICE_OK)
			{
				LogMessage("Failed to detect the peak wavelength of a lamp.\n", true);
				return ret;
			}
		}

		snprintf(nameBuf, sizeof(nameBuf) - 1, "Lamp %d peak wavelength", i + 1);
		snprintf(valueBuf, sizeof(valueBuf), "%ld", data);
		CreateProperty(nameBuf, valueBuf, MM::Integer, true);

		if (lampExists_[i])
		{
			ret = GetSetting(deviceAddress_, i + 1, "lamp.wavelength.fwhm", data);
			if (ret != DEVICE_OK)
			{
				LogMessage("Failed to detect the full-width half-magnitude of a lamp.\n", true);
				return ret;
			}
		}

		snprintf(nameBuf, sizeof(nameBuf) - 1, "Lamp %d full width half magnitude", i + 1);
		snprintf(valueBuf, sizeof(valueBuf), "%ld", data);
		CreateProperty(nameBuf, valueBuf, MM::Integer, true);

		analogData = 0.0;
		if (lampExists_[i])
		{
			ret = GetSetting(deviceAddress_, i + 1, "lamp.flux", analogData);
			if (ret != DEVICE_OK)
			{
				LogMessage("Failed to read flux of a lamp.\n", true);
				return ret;
			}
		}

		currentFlux_[i] = analogData;

		if (lampExists_[i])
		{
			ret = GetSetting(deviceAddress_, i + 1, "lamp.flux.max", analogData);
			if (ret != DEVICE_OK)
			{
				LogMessage("Failed to read max flux of a lamp.\n", true);
				return ret;
			}
		}

		maxFlux_[i] = analogData;

		snprintf(nameBuf, sizeof(nameBuf) - 1, "Lamp %d intensity", i + 1);
		action = new CPropertyActionEx(this, &Illuminator::IntensityGetSet, i);
		CreateProperty(nameBuf, "0", MM::Float, false, action);
		SetPropertyLimits(nameBuf, 0.0, 100.0);
	}

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (ret == DEVICE_OK)
	{
		initialized_ = true;
		return DEVICE_OK;
	}
	else
	{
		return ret;
	}
}


int Illuminator::Shutdown()
{
   LogMessage("Illuminator::Shutdown\n", true);

   if (initialized_)
   {
      initialized_ = false;
   }

   if (NULL != currentFlux_)
   {
	   delete[] currentFlux_;
	   currentFlux_ = NULL;
   }

   if (NULL != maxFlux_)
   {
	   delete[] maxFlux_;
	   maxFlux_ = NULL;
   }

   if (NULL != lampExists_)
   {
	   delete[] lampExists_;
	   lampExists_ = NULL;
   }

   if (NULL != lampIsOn_)
   {
	   delete[] lampIsOn_;
	   lampIsOn_ = NULL;
   }

   numLamps_ = 0;

   return DEVICE_OK;
}


bool Illuminator::Busy()
{
   this->LogMessage("Illuminator::Busy\n", true);
   return IsBusy(deviceAddress_);
}


int Illuminator::SetOpen(bool open)
{
	if (open)
	{
		if (canUseDeviceLampOnCommand_)
		{
			int ret = Command(deviceAddress_, 0, "lamp on");
			if (ret == DEVICE_OK)
			{
				isOpen_ = true;
				for (int i = 0; i < numLamps_; ++i)
				{
					if (lampExists_[i])
					{
						lampIsOn_[i] = true;
					}
				}
			}

			return ret;
		}
		else
		{
			for (int i = 0; i < numLamps_; ++i)
			{
				if (lampExists_[i] && !lampIsOn_[i] && (currentFlux_[i] > 0.0))
				{
					int ret = Command(deviceAddress_, i + 1, "lamp on");
					if (ret != DEVICE_OK)
					{
						return ret;
					}

					lampIsOn_[i] = true;
				}
			}

			isOpen_ = true;
			return DEVICE_OK;
		}
	}
	else
	{
		int ret = Command(deviceAddress_, 0, "lamp off"); // Always works at device scope.
		if (ret == DEVICE_OK)
		{
			isOpen_ = false;
			for (int i = 0; i < numLamps_; ++i)
			{
				if (lampExists_[i])
				{
					lampIsOn_[i] = false;
				}
			}
		}

		return ret;
	}
}


int Illuminator::GetOpen(bool& open)
{
	int ret = RefreshLampStatus();
	open = isOpen_;
	return ret;
}


int Illuminator::Fire(double deltaT)
{
	(void)deltaT;
	return DEVICE_UNSUPPORTED_COMMAND;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int Illuminator::PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ostringstream os;
   os << "Illuminator::PortGetSet(" << pProp << ", " << eAct << ")\n";
   this->LogMessage(os.str().c_str(), false);

   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
			resetConnection();
      }

      pProp->Get(port_);
   }

   return DEVICE_OK;
}


int Illuminator::DeviceAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   this->LogMessage("Illuminator::DeviceAddressGetSet\n", true);

   if (eAct == MM::AfterSet)
   {
      pProp->Get(deviceAddress_);
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceAddress_);
   }

   return DEVICE_OK;
}


int Illuminator::IntensityGetSet(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
   this->LogMessage("Illuminator::IntensityGetSet\n", true);

   if (eAct == MM::BeforeGet)
   {
      double flux = 1;
      if (initialized_ && lampExists_[index])
      {
         int ret = GetSetting(deviceAddress_, index + 1, "lamp.flux", flux);
         if (ret != DEVICE_OK)
         {
			return ret;
         }

		 currentFlux_[index] = flux;
      }

      pProp->Set(100.0 * flux / maxFlux_[index]);
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_ && lampExists_[index])
      {
		double intensity;
		pProp->Get(intensity);
		double flux = (maxFlux_[index] * intensity) / 100.0;

        int ret = SetSetting(deviceAddress_, index + 1, "lamp.flux", flux, 3);
        if (ret != DEVICE_OK)
        {
            return ret;
        }

		currentFlux_[index] = flux;

		// Turn the lamp on if the shutter is open.
		if (isOpen_ && !lampIsOn_[index] && (flux > 0.0))
		{
			ret = Command(deviceAddress_, index + 1, "lamp on");
			if (ret != DEVICE_OK)
			{
				return ret;
			}

			lampIsOn_[index] = true;
		}
      }
   }

   return DEVICE_OK;
}


///// Private methods /////

int Illuminator::RefreshLampStatus()
{
	std::vector<double> statusAll;
	int ret = GetSettings(deviceAddress_, 0, "lamp.status", statusAll);
	int secondaryResult = DEVICE_OK;
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (statusAll.empty())
	{
		canUseDeviceLampOnCommand_ = false;
		for (int i = 0; i < numLamps_; i++)
		{
			lampExists_[i] = false;
			lampIsOn_[i] = false;
		}

		return DEVICE_OK;
	}

	isOpen_ = false;
	canUseDeviceLampOnCommand_ = true;

	for (int index = 0; index < statusAll.size(); index++)
	{
		auto status = statusAll[index];
		if (status != status) // Inactive axis.
		{
			lampExists_[index] = false;
			lampIsOn_[index] = false;
		}
		else
		{
			switch (static_cast<int>(status))
			{
				case 1:
					lampExists_[index] = true;
					lampIsOn_[index] = false;
					break;
				case 2:
					lampExists_[index] = true;
					lampIsOn_[index] = true;
					isOpen_ = true;
					break;
				case 3:
					secondaryResult = ERR_LAMP_OVERHEATED;
					lampExists_[index] = true;
					lampIsOn_[index] = false;
					break;
				default: // Disconnected or unsupported state.
					lampExists_[index] = false;
					lampIsOn_[index] = false;
					canUseDeviceLampOnCommand_ = false;
					break;
			}
		}
	}

	return secondaryResult;
}
