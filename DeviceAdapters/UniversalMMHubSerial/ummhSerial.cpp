///////////////////////////////////////////////////////////////////////////////
// FILE:          ummhSerial.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implementation of the universal hardware hub
//                that uses a serial port for communication
//                
// COPYRIGHT:     Artem Melnykov, 2021-2022
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Artem Melnykov, melnykov.artem at gmail.com, 2021
//                

#include "ummhSerial.h"
#include "ummhreserved.h"

using namespace std;

// External names used by the rest of the system
// to load particular device from the device adapter dll
const char* g_HubDeviceName = "UniversalMMHubSerial";
const char* g_HubDeviceDescription = "Universal hardware hub (serial)";

vector<mmdevicedescription> deviceDescriptionList;
UniHub* hub_ = 0;

// split a string ('line') into a vector of strings based on the provided separator
vector<string> SplitStringIntoWords(std::string line, char sep)
{
	// split the string into words
	vector<string> retwords;
	size_t pos1 = 0, pos2 = 0;
	pos1 = (size_t)-1;
	pos2 = line.find(sep, pos1 + 1);
	if (pos2 == pos1) {
		retwords.push_back(line);
		return retwords;
	}
	while (pos2 != string::npos)
	{
		retwords.push_back(line.substr(pos1 + 1, pos2 - (pos1 + 1)));
		pos1 = pos2;
		pos2 = line.find(sep, pos1 + 1);
	}
	retwords.push_back(line.substr(pos1 + 1, line.length() - 1));
	return retwords;
}


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_HubDeviceName, MM::HubDevice, g_HubDeviceDescription);
	for (unsigned ii = 0; ii< deviceDescriptionList.size(); ii++) {
		mmdevicedescription d = deviceDescriptionList.at(ii);
		if (!d.isValid) continue; // skip this device if deviceDescription is not properly defined
		if (strcmp(MM::g_Keyword_CoreShutter, d.type.c_str()) == 0) {
			RegisterDevice(d.name.c_str(), MM::ShutterDevice, d.description.c_str());
		}
		else if (strcmp(MM::g_Keyword_State, d.type.c_str()) == 0) {
			RegisterDevice(d.name.c_str(), MM::StateDevice, d.description.c_str());
		}
		else if (strcmp("Stage", d.type.c_str()) == 0) {
			RegisterDevice(d.name.c_str(), MM::StageDevice, d.description.c_str());
		}
		else if (strcmp(MM::g_Keyword_CoreXYStage, d.type.c_str()) == 0) {
			RegisterDevice(d.name.c_str(), MM::XYStageDevice, d.description.c_str());
		}
		else if (strcmp("Generic", d.type.c_str()) == 0) {
			RegisterDevice(d.name.c_str(), MM::GenericDevice, d.description.c_str());
		}
	}
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0) {
	   return 0;
   }
   else if (strcmp(deviceName, g_HubDeviceName) == 0) // create the hub
   {
	   hub_ = new UniHub();
	   return hub_;
   }
   else
   {
	   if (strncmp(deviceName, MM::g_Keyword_CoreShutter, strlen(MM::g_Keyword_CoreShutter)) == 0) {
		   return new UmmhShutter(deviceName);
	   }
	   else if (strncmp(deviceName, MM::g_Keyword_State, strlen(MM::g_Keyword_State)) == 0) {
		   return new UmmhStateDevice(deviceName);
	   }
	   else if (strncmp(deviceName, "Stage", strlen("Stage")) == 0) {
		   return new UmmhStage(deviceName);
	   }
	   else if (strncmp(deviceName, MM::g_Keyword_CoreXYStage, strlen(MM::g_Keyword_CoreXYStage)) == 0) {
		   return new UmmhXYStage(deviceName);
	   }
	   else if (strncmp(deviceName, "Generic", strlen("Generic")) == 0) {
		   return new UmmhGeneric(deviceName);
	   }
	   else {
		   // ...supplied name not recognized
		   return 0;
	   }
   }

}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

// ******* UniversalMMHubSerial *****************
// ************** start *************************
UniHub::UniHub() :
	busy_(false),
	initialized_(false),
	error_(0),
	port_("Click to select..."),
	thr_(0)
{
	stopbusythread_ = false;
	deviceDescriptionList.clear();

	InitializeDefaultErrorMessages();

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &UniHub::OnPort);
	// *** serial communication ***
	// g_Keyword_Port is used to create an instance of SerialManager
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);		
}
 
UniHub::~UniHub()
{
	Shutdown();
}

int UniHub::Initialize()
{	
	if (initialized_) return DEVICE_OK;
	int ret;

	// Name
	ret = CreateStringProperty(MM::g_Keyword_Name, g_HubDeviceName, true);
	if (DEVICE_OK != ret) return ret;

	// Description
	ret = CreateStringProperty(MM::g_Keyword_Description, g_HubDeviceDescription, true);
	if (DEVICE_OK != ret) return ret;

	// Error reporter
	CPropertyAction* pAct;
	pAct = new CPropertyAction(this, &UniHub::OnError);
	ret = CreateIntegerProperty("Error",0,false,pAct,false);
	if (ret != DEVICE_OK) return ret;
	ret = CreateStringProperty("Error Description", "none", false);
	if (ret != DEVICE_OK) return ret;

	Sleep(1000); // give serial port time to initialize
	PurgeComPort(port_.c_str());
	ret = PopulateDeviceDescriptionList();
	if (ret != DEVICE_OK) return ret;

	stringstream sss;
	sss << "Device description list length = " << deviceDescriptionList.size();
	LogMessage(sss.str().c_str(), true);
	for (int ii = 0; ii < deviceDescriptionList.size(); ii++) {
		mmdevicedescription dd = deviceDescriptionList.at(ii);
		LogMessage(dd.name, dd.isValid);
		LogMessage(dd.type, dd.isValid);
		if (!dd.isValid) LogMessage(dd.reasonWhyInvalid, false);
	}

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	DetectInstalledDevices();

	// create a thread for updating busy status
	thr_ = new BusyThread(this);
	thr_->activate();

	initialized_ = true;

	return DEVICE_OK;
}

int UniHub::Shutdown()
{
	if (initialized_)
	{
		stopbusythread_ = true;
		if (thr_!=0) thr_->wait();
		initialized_ = false;
	}
	deviceDescriptionList.clear();
	hub_ = 0;
	return DEVICE_OK;
}

void UniHub::GetName(char* pName) const
{
	CDeviceUtils::CopyLimitedString(pName, g_HubDeviceName);
}

int UniHub::DetectInstalledDevices()
{  
   ClearInstalledDevices();

   // make sure this method is called before we look for available devices
   InitializeModuleData();

   char hubName[MM::MaxStrLength];
   GetName(hubName); // this device name
   for (unsigned i = 0; i < GetNumberOfDevices(); i++)
   { 
      char deviceName[MM::MaxStrLength];
      bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
      if (success && (strcmp(hubName, deviceName) != 0))
      {
         MM::Device* pDev = CreateDevice(deviceName);
         AddInstalledDevice(pDev);
      }
   }
   return DEVICE_OK; 
}

bool UniHub::Busy()
{
	return busy_;
}

int UniHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		string  newportvalue;
		pProp->Get(newportvalue);
		if (strcmp(port_.c_str(), newportvalue.c_str()) == 0) return DEVICE_OK;
		if (initialized_)
		{
			// revert
			pProp->Set(port_.c_str());
			return DEVICE_CAN_NOT_SET_PROPERTY;// port change forbidden
		}
		pProp->Get(port_);
	}
	return DEVICE_OK;
}

void UniHub::SetBusy(string devicename, bool val) {
	MM::Device* pDevice = GetDevice(devicename.c_str());
	if (pDevice->GetType() == MM::DeviceType::ShutterDevice) {
		UmmhShutter* p = static_cast<UmmhShutter*>(pDevice);
		p->SetBusy(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::StateDevice) {
		UmmhStateDevice* p = static_cast<UmmhStateDevice*>(pDevice);
		p->SetBusy(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::StageDevice) {
		UmmhStage* p = static_cast<UmmhStage*>(pDevice);
		p->SetBusy(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* p = static_cast<UmmhXYStage*>(pDevice);
		p->SetBusy(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::GenericDevice) {
		UmmhGeneric* p = static_cast<UmmhGeneric*>(pDevice);
		p->SetBusy(val);
	}
}

MM::MMTime UniHub::GetTimeout(string devicename) {
	MM::Device* pDevice = GetDevice(devicename.c_str());
	if (pDevice->GetType() == MM::DeviceType::ShutterDevice) {
		UmmhShutter* p = static_cast<UmmhShutter*>(pDevice);
		return p->GetTimeout();
	} else if (pDevice->GetType() == MM::DeviceType::StateDevice) {
		UmmhStateDevice* p = static_cast<UmmhStateDevice*>(pDevice);
		return p->GetTimeout();
	}
	else if (pDevice->GetType() == MM::DeviceType::StageDevice) {
		UmmhStage* p = static_cast<UmmhStage*>(pDevice);
		return p->GetTimeout();
	}
	else if (pDevice->GetType() == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* p = static_cast<UmmhXYStage*>(pDevice);
		return p->GetTimeout();
	}
	else if (pDevice->GetType() == MM::DeviceType::GenericDevice) {
		UmmhGeneric* p = static_cast<UmmhGeneric*>(pDevice);
		return p->GetTimeout();
	}
	return 0;
}

void UniHub::SetTimeout(string devicename, MM::MMTime val) {
	MM::Device* pDevice = GetDevice(devicename.c_str());
	if (pDevice->GetType() == MM::DeviceType::ShutterDevice) {
		UmmhShutter* p = static_cast<UmmhShutter*>(pDevice);
		p->SetTimeout(val);
	} else if (pDevice->GetType() == MM::DeviceType::StateDevice) {
		UmmhStateDevice* p = static_cast<UmmhStateDevice*>(pDevice);
		p->SetTimeout(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::StageDevice) {
		UmmhStage* p = static_cast<UmmhStage*>(pDevice);
		p->SetTimeout(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* p = static_cast<UmmhXYStage*>(pDevice);
		p->SetTimeout(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::GenericDevice) {
		UmmhGeneric* p = static_cast<UmmhGeneric*>(pDevice);
		p->SetTimeout(val);
	}
}

MM::MMTime UniHub::GetLastCommandTime(string devicename) {
	MM::Device* pDevice = GetDevice(devicename.c_str());
	if (pDevice->GetType() == MM::DeviceType::ShutterDevice) {
		UmmhShutter* p = static_cast<UmmhShutter*>(pDevice);
		return p->GetLastCommandTime();
	} else if (pDevice->GetType() == MM::DeviceType::StateDevice) {
		UmmhStateDevice* p = static_cast<UmmhStateDevice*>(pDevice);
		return p->GetLastCommandTime();
	}
	else if (pDevice->GetType() == MM::DeviceType::StageDevice) {
		UmmhStage* p = static_cast<UmmhStage*>(pDevice);
		return p->GetLastCommandTime();
	}
	else if (pDevice->GetType() == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* p = static_cast<UmmhXYStage*>(pDevice);
		return p->GetLastCommandTime();
	}
	else if (pDevice->GetType() == MM::DeviceType::GenericDevice) {
		UmmhGeneric* p = static_cast<UmmhGeneric*>(pDevice);
		return p->GetLastCommandTime();
	}
	return 0;
}

void UniHub::SetLastCommandTime(string devicename, MM::MMTime val) {
	MM::Device* pDevice = GetDevice(devicename.c_str());
	if (pDevice->GetType() == MM::DeviceType::ShutterDevice) {
		UmmhShutter* p = static_cast<UmmhShutter*>(pDevice);
		p->SetLastCommandTime(val);
	} else if (pDevice->GetType() == MM::DeviceType::StateDevice) {
		UmmhStateDevice* p = static_cast<UmmhStateDevice*>(pDevice);
		p->SetLastCommandTime(val);
	} else if (pDevice->GetType() == MM::DeviceType::StageDevice) {
		UmmhStage* p = static_cast<UmmhStage*>(pDevice);
		p->SetLastCommandTime(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* p = static_cast<UmmhXYStage*>(pDevice);
		p->SetLastCommandTime(val);
	}
	else if (pDevice->GetType() == MM::DeviceType::GenericDevice) {
		UmmhGeneric* p = static_cast<UmmhGeneric*>(pDevice);
		p->SetLastCommandTime(val);
	}
}

int UniHub::ReportToDevice(string devicename, string command, vector<string> vals) {
	int ret = DEVICE_OK;
	// find device
	MM::Device* pDevice = GetDevice(devicename.c_str());
	// find its type
	MM::DeviceType type = pDevice->GetType();	

	// Shutter response
	if (type == MM::DeviceType::ShutterDevice) {
		UmmhShutter* pShutter = static_cast<UmmhShutter*>(pDevice);
		int index = GetDeviceIndexFromName(devicename);
		mmdevicedescription d = deviceDescriptionList.at(index);
		// report timeout change
		if (command.compare(ummhwords::timeout)==0) {
			double v = atof(vals[0].c_str())*1000;
			pShutter->SetLastCommandTime(GetCurrentMMTime());
			pShutter->SetTimeout(MM::MMTime(v));
			return DEVICE_OK;
		}
		// report a method change
		mmmethoddescription md;
		for (int ii = 0; ii < d.methods.size(); ii++) { // find the method that corresponds to this command
			md = d.methods.at(ii);
			if (md.command.compare(command) == 0) {
				// found this command
				if (md.method.compare(ummhwords::set_open)==0 || md.method.compare(ummhwords::get_open)==0) {
					pShutter->SetUpdating(true);
					ret = pShutter->SetOpen(atoi(vals[0].c_str()));
				}
				return ret;
			}
		}
		// report a property change
		mmpropertydescription pd;
		for (int ii = 0; ii < d.properties.size(); ii++) {
			int jj;
			pd = d.properties.at(ii);
			if (pd.cmdAction.compare(command) == 0) {
				// found this property
				if (pd.type == MM::PropertyType::String) {
					for (jj = 0; jj < pd.allowedValues.size(); jj++) {
						if (pd.allowedValues.at(jj).compare(vals[0]) == 0) {
							deviceDescriptionList.at(index).properties.at(ii).valueString = vals[0];
							break;
						}
					}
					if (jj == pd.allowedValues.size()) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
				}
				else if (pd.type == MM::PropertyType::Integer) {
					int value = atoi(vals[0].c_str());
					if (value<pd.lowerLimitInteger || value>pd.upperLimitInteger) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueInteger = value;
					}
				}
				else if (pd.type == MM::PropertyType::Float) {
					float value = (float)atof(vals[0].c_str());
					if (value<pd.lowerLimitFloat || value>pd.upperLimitFloat) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueFloat = value;
					}
				}
				ret = OnPropertyChanged(pd.name.c_str(), vals[0].c_str());
				return ret;
			}
		}
		return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_not_recognized); // if method or property was not found
	}

	// State response
	if (type == MM::DeviceType::StateDevice) {
		UmmhStateDevice* pState = static_cast<UmmhStateDevice*>(pDevice);
		int index = GetDeviceIndexFromName(devicename);
		mmdevicedescription d = deviceDescriptionList.at(index);
		// report timeout change
		if (command.compare(ummhwords::timeout) == 0) {
			double v = atof(vals[0].c_str()) * 1000;
			pState->SetLastCommandTime(GetCurrentMMTime());
			pState->SetTimeout(MM::MMTime(v));
			return DEVICE_OK;
		}
		// report a property change
		mmpropertydescription pd;
		for (int ii = 0; ii < d.properties.size(); ii++) {
			int jj;
			pd = d.properties.at(ii);
			if (pd.cmdAction.compare(command) == 0) {
				// found this property
				if (pd.type == MM::PropertyType::String) {
					for (jj = 0; jj < pd.allowedValues.size(); jj++) {
						if (pd.allowedValues.at(jj).compare(vals[0]) == 0) {
							deviceDescriptionList.at(index).properties.at(ii).valueString = vals[0];
							break;
						}
					}
					if (jj == pd.allowedValues.size()) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
				}
				else if (pd.type == MM::PropertyType::Integer) {
					int value = atoi(vals[0].c_str());
					if (value<pd.lowerLimitInteger || value>pd.upperLimitInteger) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueInteger = value;
					}
				}
				else if (pd.type == MM::PropertyType::Float) {
					float value = (float)atof(vals[0].c_str());
					if (value<pd.lowerLimitFloat || value>pd.upperLimitFloat) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueFloat = value;
					}
				}
				ret = OnPropertyChanged(pd.name.c_str(), vals[0].c_str());
				return ret;
			}
		}
		return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_not_recognized); // if method or property was not found
	}

	// Stage response
	if (type == MM::DeviceType::StageDevice) {
		UmmhStage* pStage = static_cast<UmmhStage*>(pDevice);
		int index = GetDeviceIndexFromName(devicename);
		mmdevicedescription d = deviceDescriptionList.at(index);
		// report timeout change
		if (command.compare(ummhwords::timeout) == 0) {
			double v = atof(vals[0].c_str()) * 1000;
			pStage->SetLastCommandTime(GetCurrentMMTime());
			pStage->SetTimeout(MM::MMTime(v));
			return DEVICE_OK;
		}
		// report a method change
		mmmethoddescription md;
		for (int ii = 0; ii < d.methods.size(); ii++) { // find the method that corresponds to this command
			md = d.methods.at(ii);
			if (md.command.compare(command) == 0) {
				// found this command
				if (md.method.compare(ummhwords::set_position_um) == 0 || md.method.compare(ummhwords::get_position_um) == 0 || 
					md.method.compare(ummhwords::home) == 0 || md.method.compare(ummhwords::stop) == 0) {
					pStage->SetUpdating(true);
					ret = pStage->SetPositionUm(atof(vals[0].c_str()));
				}
				return ret;
			}
		}
		// report a property change
		mmpropertydescription pd;
		for (int ii = 0; ii < d.properties.size(); ii++) {
			int jj;
			pd = d.properties.at(ii);
			if (pd.cmdAction.compare(command) == 0) {
				// found this property
				if (pd.type == MM::PropertyType::String) {
					for (jj = 0; jj < pd.allowedValues.size(); jj++) {
						if (pd.allowedValues.at(jj).compare(vals[0]) == 0) {
							deviceDescriptionList.at(index).properties.at(ii).valueString = vals[0];
							break;
						}
					}
					if (jj == pd.allowedValues.size()) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
				}
				else if (pd.type == MM::PropertyType::Integer) {
					int value = atoi(vals[0].c_str());
					if (value<pd.lowerLimitInteger || value>pd.upperLimitInteger) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueInteger = value;
					}
				}
				else if (pd.type == MM::PropertyType::Float) {
					float value = (float)atof(vals[0].c_str());
					if (value<pd.lowerLimitFloat || value>pd.upperLimitFloat) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueFloat = value;
					}
				}
				ret = OnPropertyChanged(pd.name.c_str(), vals[0].c_str());
				return ret;
			}
		}
		return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_not_recognized); // if method or property was not found
	}

	// XYStage response
	if (type == MM::DeviceType::XYStageDevice) {
		UmmhXYStage* pXYStage = static_cast<UmmhXYStage*>(pDevice);
		int index = GetDeviceIndexFromName(devicename);
		mmdevicedescription d = deviceDescriptionList.at(index);
		// report timeout change
		if (command.compare(ummhwords::timeout) == 0) {
			double v = atof(vals[0].c_str()) * 1000;
			pXYStage->SetLastCommandTime(GetCurrentMMTime());
			pXYStage->SetTimeout(MM::MMTime(v));
			return DEVICE_OK;
		}
		// report a method change
		mmmethoddescription md;
		for (int ii = 0; ii < d.methods.size(); ii++) { // find the method that corresponds to this command
			md = d.methods.at(ii);
			if (md.command.compare(command) == 0) {
				// found this command
				if (md.method.compare(ummhwords::set_position_um) == 0 || md.method.compare(ummhwords::get_position_um) == 0 ||
					md.method.compare(ummhwords::home) == 0 || md.method.compare(ummhwords::stop) == 0) {
					pXYStage->SetUpdating(true);
					ret = pXYStage->SetPositionUm(atof(vals[0].c_str()), atof(vals[1].c_str()));
				}
				return ret;
			}
		}
		// report a property change
		mmpropertydescription pd;
		for (int ii = 0; ii < d.properties.size(); ii++) {
			int jj;
			pd = d.properties.at(ii);
			if (pd.cmdAction.compare(command) == 0) {
				// found this property
				if (pd.type == MM::PropertyType::String) {
					for (jj = 0; jj < pd.allowedValues.size(); jj++) {
						if (pd.allowedValues.at(jj).compare(vals[0]) == 0) {
							deviceDescriptionList.at(index).properties.at(ii).valueString = vals[0];
							break;
						}
					}
					if (jj == pd.allowedValues.size()) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
				}
				else if (pd.type == MM::PropertyType::Integer) {
					int value = atoi(vals[0].c_str());
					if (value<pd.lowerLimitInteger || value>pd.upperLimitInteger) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueInteger = value;
					}
				}
				else if (pd.type == MM::PropertyType::Float) {
					float value = (float)atof(vals[0].c_str());
					if (value<pd.lowerLimitFloat || value>pd.upperLimitFloat) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueFloat = value;
					}
				}
				ret = OnPropertyChanged(pd.name.c_str(), vals[0].c_str());
				return ret;
			}
		}
		return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_not_recognized); // if method or property was not found
	}

	// Generic device response
	if (type == MM::DeviceType::GenericDevice) {
		UmmhGeneric* pShutter = static_cast<UmmhGeneric*>(pDevice);
		int index = GetDeviceIndexFromName(devicename);
		mmdevicedescription d = deviceDescriptionList.at(index);
		// report timeout change
		if (command.compare(ummhwords::timeout) == 0) {
			double v = atof(vals[0].c_str()) * 1000;
			pShutter->SetLastCommandTime(GetCurrentMMTime());
			pShutter->SetTimeout(MM::MMTime(v));
			return DEVICE_OK;
		}
		// report a property change
		mmpropertydescription pd;
		for (int ii = 0; ii < d.properties.size(); ii++) {
			int jj;
			pd = d.properties.at(ii);
			if (pd.cmdAction.compare(command) == 0) {
				// found this property
				if (pd.type == MM::PropertyType::String) {
					for (jj = 0; jj < pd.allowedValues.size(); jj++) {
						if (pd.allowedValues.at(jj).compare(vals[0]) == 0) {
							deviceDescriptionList.at(index).properties.at(ii).valueString = vals[0];
							break;
						}
					}
					if (jj == pd.allowedValues.size()) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
				}
				else if (pd.type == MM::PropertyType::Integer) {
					int value = atoi(vals[0].c_str());
					if (value<pd.lowerLimitInteger || value>pd.upperLimitInteger) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueInteger = value;
					}
				}
				else if (pd.type == MM::PropertyType::Float) {
					float value = (float)atof(vals[0].c_str());
					if (value<pd.lowerLimitFloat || value>pd.upperLimitFloat) {
						return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_value_not_allowed);
					}
					else {
						deviceDescriptionList.at(index).properties.at(ii).valueFloat = value;
					}
				}
				ret = OnPropertyChanged(pd.name.c_str(), vals[0].c_str());
				return ret;
			}
		}
		return ReportErrorForDevice(devicename, command, vals, ummherrors::adp_device_command_not_recognized); // if method or property was not found
	}

	return DEVICE_ERR;
}

int UniHub::ReportErrorForDevice(string devicename, string command, vector<string> vals, int err) {
	stringstream ss;
	ss << devicename << "," << command << ",";
	for (int ii = 0; ii < vals.size(); ii++) {
		ss << vals.at(ii);
		if (ii != vals.size() - 1) ss << ",";
	}
	return WriteError(ss.str(), err);
}

int UniHub::WriteError(string addonstr, int err) {	
	stringstream ss;
	ss << "UMMH error (" << port_ << "):";
	switch (err) {
	case ummherrors::adp_communication_error:
		ss << "controller communication error (";
		break;
	case ummherrors::adp_version_mismatch:
		ss << "Version number specified by the controller is not supported by this adapter (";
		break;
	case ummherrors::adp_lost_communication:
		ss << "Lost communication with the controller (";
		break;
	case ummherrors::adp_string_not_recognized:
		ss << "Unable to parse string returned by the controller (";
		break;
	case ummherrors::adp_device_not_recognized:
		ss << "Device was not recognized (";
		break;
	case ummherrors::adp_device_command_not_recognized:
		ss << "Device command was not recognized (";
		break;
	case ummherrors::adp_device_command_value_not_allowed:
		ss << "Device command value was not recognized (";
		break;
	case ummherrors::ctr_string_not_recognized:
		ss << "Adapter sent a string that controller is unable parse (";
		break;
	case ummherrors::ctr_device_not_recognized:
		ss << "Device was not recognized by the controller (";
		break;
	case ummherrors::ctr_device_command_not_recognized:
		ss << "Device command was not recognized by the controller (";
		break;
	case ummherrors::ctr_device_command_value_not_allowed:
		ss << "Device command value not allowed by the controller (";
		break;
	case ummherrors::ctr_device_timeout:
		ss << "Controller reported timeout (";
		break;
	default:
		ss << "Unknown error (";
	}
	ss << addonstr << "); error code " << err;
	LogMessage(ss.str(), false);
	error_ = err;
	stringstream se;
	se << error_;
	OnPropertyChanged("Error", se.str().c_str());
	SetProperty("Error Description", ss.str().c_str());
	return err;
}

int UniHub::OnError(MM::PropertyBase* pProp, MM::ActionType eAct) {
	long val;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)error_);
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(val);
		error_ = val;
		if (error_ == 0) {
			SetProperty("Error Description", "none");
		}
		return DEVICE_OK;
	}
	return DEVICE_OK;
}

int UniHub::ReportTimeoutError(string name) {
	stringstream ss;
	stringstream se;
	error_ = ummherrors::adp_lost_communication;
	return WriteError(name,error_);
}

string UniHub::GetDeviceTypeFromName(string devicename) {
	string ret = string();
	for (int ii = 0; ii < deviceDescriptionList.size(); ii++) {
		mmdevicedescription d = deviceDescriptionList.at(ii);
		if (d.name.compare(devicename) == 0) {
			ret = d.type;
			break;
		}
	}
	return ret;
}

int UniHub::GetDeviceIndexFromName(string devicename) {
	int ret_index = -1;
	for (int ii = 0; ii < deviceDescriptionList.size(); ii++) {
		mmdevicedescription d = deviceDescriptionList.at(ii);
		if (d.name.compare(devicename) == 0) {
			ret_index = ii;
			break;
		}
	}
	return ret_index;
}

int UniHub::PopulateDeviceDescriptionList() {
	int ret;
	string ans;
	stringstream ssstart, ssnext;
	vector<string> words;
	vector<string> device_vecstr;
	mmdevicedescription dd;

	ssstart << ummhwords::device_list_start << ummhwords::sepEnd;
	ret = SendCommand(ssstart.str());
	ssnext << ummhwords::device_list_continue << ummhwords::sepEnd;
	ret = ReceiveAndWaitForAnswer(ans,MM::MMTime(1e6));
	while (ret==DEVICE_OK) {
		if (ans.compare(ummhwords::device_list_end)==0) break;
		words = SplitStringIntoWords(ans, ummhwords::sepSetup);
		if (words[0].compare(MM::g_Keyword_Name)==0) {
			if (device_vecstr.size() != 0) {
				dd = this->VectorstrToDeviceDescription(device_vecstr);
				//if (dd.isValid) deviceDescriptionList.push_back(dd);
				deviceDescriptionList.push_back(dd);
				// make a new device description
				device_vecstr.clear();
			}
		}
		device_vecstr.push_back(ans);
		ret = SendCommand(ssnext.str());
		if (ret != DEVICE_OK) return ret;
		ret = ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
	}
	// make the last device description
	dd = this->VectorstrToDeviceDescription(device_vecstr);
	//if (dd.isValid) deviceDescriptionList.push_back(dd);
	deviceDescriptionList.push_back(dd);

	return ret;
}

mmdevicedescription UniHub::VectorstrToDeviceDescription(vector<string> vs) {
	
	mmdevicedescription devdescr;
	vector<string> words;
	devdescr.isValid = true;
	
	for (int ii = 0; ii < vs.size(); ii++) {
		// split the string into words
		string s = vs.at(ii);
		words = SplitStringIntoWords(s,ummhwords::sepSetup);
		
		// at least two words must be present in the string
		if (words.size() < 2) {
			devdescr.isValid = false;
			devdescr.reasonWhyInvalid.append("Invalid string: ");
			devdescr.reasonWhyInvalid.append(s);
			break;
		}

		// get device type and name
		if (strcmp(words.at(0).c_str(), MM::g_Keyword_Name) == 0) {
			devdescr.name = words.at(1);
			// assign device type
			if (strncmp(words.at(1).c_str(),MM::g_Keyword_CoreShutter,strlen(MM::g_Keyword_CoreShutter))==0) {
				devdescr.type = MM::g_Keyword_CoreShutter;
			}
			else if (strncmp(words.at(1).c_str(), MM::g_Keyword_State, strlen(MM::g_Keyword_State)) == 0) {
				devdescr.type = MM::g_Keyword_State;
			}
			else if (strncmp(words.at(1).c_str(), "Stage", strlen("Stage")) == 0) {
				devdescr.type = "Stage";
			}
			else if (strncmp(words.at(1).c_str(), MM::g_Keyword_CoreXYStage, strlen(MM::g_Keyword_CoreXYStage)) == 0) {
				devdescr.type = MM::g_Keyword_CoreXYStage;
			}
			else if (strncmp(words.at(1).c_str(), "Generic", strlen("Generic")) == 0) {
				devdescr.type = MM::GenericDevice;
			}
			else {
				devdescr.isValid = false;
				devdescr.reasonWhyInvalid.append("Unable to determine device type for ");
				devdescr.reasonWhyInvalid.append(s);
				break;
			}
		}
		// get device description
		else if (strcmp(words.at(0).c_str(), MM::g_Keyword_Description) == 0) {
			devdescr.description = words.at(1);
		}
		// get device timeout
		else if (strcmp(words.at(0).c_str(), ummhwords::timeout) == 0) {
			devdescr.timeout = MM::MMTime(1000*atof(words.at(1).c_str()));
		}
		// get commands
		else if (strcmp(words.at(0).c_str(), ummhwords::cmd) == 0) { // a standard command
			mmmethoddescription cd;
			cd.method = words.at(1);
			cd.command = words.at(2);
			devdescr.methods.push_back(cd);
		}
		// get properties
		else if (words.at(0).find(ummhwords::prop)==0 ) {
			if (words.at(0).find(ummhwords::act) == string::npos) { // not an action property
				if (words.size() != 5) {
					devdescr.isValid = false;
					devdescr.reasonWhyInvalid.append("Invalid property: ");
					devdescr.reasonWhyInvalid.append(s);
					break;
				}
				mmpropertydescription pd;
				pd.isAction = false;
				pd.name = words.at(1);
				if (strcmp(words.at(0).c_str(), ummhwords::prop_str) == 0) { // string property
					pd.type = MM::PropertyType::String;
					pd.valueString = words.at(2);
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(4), ummhwords::sepWithin);
						pd.allowedValues = valuelist;
					}
					else {
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status: ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else if (strcmp(words.at(0).c_str(), ummhwords::prop_float) == 0) { // float property
					pd.type = MM::PropertyType::Float;
					pd.valueFloat = stof(words.at(2));
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(4),ummhwords::sepWithin);
						if (valuelist.size() != 2) {
							devdescr.isValid = false;
							devdescr.reasonWhyInvalid.append("Unable to determine property limits: ");
							devdescr.reasonWhyInvalid.append(s);
							break;
						}
						pd.lowerLimitFloat = stof(valuelist.at(0));
						pd.upperLimitFloat = stof(valuelist.at(1));
					}
					else {
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status: ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else if (strcmp(words.at(0).c_str(), ummhwords::prop_int) == 0) { // integer property
					pd.type = MM::PropertyType::Integer;
					pd.valueInteger = stoi(words.at(2));
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(4), ummhwords::sepWithin);
						if (valuelist.size() != 2) {
							devdescr.isValid = false;
							devdescr.reasonWhyInvalid.append("Unable to determine property limits: ");
							devdescr.reasonWhyInvalid.append(s);
							break;
						}
						pd.lowerLimitInteger = stoi(valuelist.at(0));
						pd.upperLimitInteger = stoi(valuelist.at(1));
					}
					else {
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status: ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else { // unknown property type
					devdescr.isValid = false;
					devdescr.reasonWhyInvalid.append("Unable to determine property type: ");
					devdescr.reasonWhyInvalid.append(s);
					break;
				}
				devdescr.properties.push_back(pd);
			}
			else { // action property
				if (words.size() != 7) {
					devdescr.isValid = false;
					devdescr.reasonWhyInvalid.append("Invalid property: ");
					devdescr.reasonWhyInvalid.append(s);
					break;
				}
				mmpropertydescription pd;
				pd.isAction = true;
				pd.name = words.at(1); // name
				pd.cmdAction = words.at(4); // command
				if (strcmp(words.at(5).c_str(), ummhwords::wtrue) == 0) { // pre-initialization = true
					pd.isPreini = true;
				}
				else if (strcmp(words.at(5).c_str(), ummhwords::wfalse) == 0) { // pre-initialization = false
					pd.isPreini = false;
				}
				if (strcmp(words.at(0).c_str(), ummhwords::prop_str_act) == 0) { // string property
					pd.type = MM::PropertyType::String;
					pd.valueString = words.at(2);
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(6), ummhwords::sepWithin);
						pd.allowedValues = valuelist;
					}
					else {
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else if (strcmp(words.at(0).c_str(), ummhwords::prop_float_act) == 0) { // float property
					pd.type = MM::PropertyType::Float;
					pd.valueFloat = stof(words.at(2));
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(6), ummhwords::sepWithin);
						if (valuelist.size() != 2) {
							devdescr.isValid = false;
							devdescr.reasonWhyInvalid.append("Unable to determine property limits: ");
							devdescr.reasonWhyInvalid.append(s);
							break;
						}
						pd.lowerLimitFloat = stof(valuelist.at(0));
						pd.upperLimitFloat = stof(valuelist.at(1));
					}
					else { // unknown property type
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status: ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else if (strcmp(words.at(0).c_str(), ummhwords::prop_int_act) == 0) { // integer property
					pd.type = MM::PropertyType::Integer;
					pd.valueInteger = stoi(words.at(2));
					if (strcmp(words.at(3).c_str(), ummhwords::wtrue) == 0) { // read only = true
						pd.isReadOnly = true;
					}
					else if (strcmp(words.at(3).c_str(), ummhwords::wfalse) == 0) { // read only = false
						pd.isReadOnly = false;
						vector<string> valuelist = SplitStringIntoWords(words.at(6), ummhwords::sepWithin);
						if (valuelist.size() != 2) {
							devdescr.isValid = false;
							devdescr.reasonWhyInvalid.append("Unable to determine property limits: ");
							devdescr.reasonWhyInvalid.append(s);
							break;
						}
						pd.lowerLimitInteger = stoi(valuelist.at(0));
						pd.upperLimitInteger = stoi(valuelist.at(1));
					}
					else { // unknown property type
						devdescr.isValid = false;
						devdescr.reasonWhyInvalid.append("Unable to determine read-only status: ");
						devdescr.reasonWhyInvalid.append(s);
						break;
					}
				}
				else { // unknown property type
					devdescr.isValid = false;
					devdescr.reasonWhyInvalid.append("Unable to determine property type: ");
					devdescr.reasonWhyInvalid.append(s);
					break;
				}
				devdescr.properties.push_back(pd);
			}
		}


	}
	
	return devdescr;

}

string UniHub::ConvertMethodToCommand(string deviceName, string methodName) {
	// find the appropriate command for this method
	int index = GetDeviceIndexFromName(deviceName);
	mmdevicedescription d = deviceDescriptionList.at(index);
	mmmethoddescription cd;
	for (int ii = 0; ii < d.methods.size(); ii++) {
		cd = d.methods.at(ii);
		if (cd.method.compare(methodName) == 0) {
			return cd.command;
		}
	}
	return string();
}

int UniHub::MakeAndSendOutputCommand(string devicename, string command, vector<string> values) {
	stringstream ss;
	ss << devicename << ummhwords::sepOut << command << ummhwords::sepOut;
	for (int ii=0; ii<values.size()-1; ii++) {
		ss << values.at(ii) << ummhwords::sepWithin;
	}
	ss << values.at(values.size()-1) << ummhwords::sepEnd;
	int ret = SendCommand(ss.str());
	return ret;
}

int UniHub::SendCommand(string cmd) {
	string ans = string(); // not actually used
	int ret = SerialCommunication(ummhflags::serial_out, cmd, ans);
	return ret;
}

int UniHub::ReceiveAnswer(string& ans) {
	string cmd = string(); // not actually used
	int ret = SerialCommunication(ummhflags::serial_in, cmd, ans);
	return ret;
}

int UniHub::ReceiveAndWaitForAnswer(string& ans, MM::MMTime timeout) {
	string cmd = string(); // not actually used
	string temp;
	int ret = SerialCommunication(ummhflags::serial_in, cmd, temp);
	ans = temp;
	MM::MMTime commandtime = GetCurrentMMTime();
	MM::MMTime interval;
	interval = GetCurrentMMTime() - commandtime;
	while (ret != DEVICE_OK && interval < timeout) {
		ret = SerialCommunication(ummhflags::serial_in, cmd, ans);
		ans.append(temp);
		interval = GetCurrentMMTime() - commandtime;
	}
	return ret;
}

// *** serial communication ***
int UniHub::SerialCommunication(char inorout, string cmd, string& ans) {

	MMThreadGuard(this->executeLock_); // thread lock
	if (inorout == ummhflags::serial_out) { // send command and exit
		// MM implementation
		int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), "");
		return ret;
	}
	else if (inorout == ummhflags::serial_in) { // try to receive a coomand and exit regardless of the status
		// MM implementation
		int ret;
		char term[2];
		term[0] = ummhwords::sepEnd;
		term[1] = '\0';
		ret = GetSerialAnswer(port_.c_str(), term, ans);
		return ret;
	}
	else {
		return DEVICE_SERIAL_COMMAND_FAILED;
	}

}

int UniHub::CheckIncomingCommand(vector<string> vs) {
	
	// check overall format
	if (vs.size() != 3) return ummherrors::adp_string_not_recognized;
	
	// check if device exists
	string name = vs.at(0);
	if (GetDevice(name.c_str()) == 0) return ummherrors::adp_device_not_recognized;

	// command checking is done by ReportToDevice

	// command value checking is done by ReportToDevice for properties and 
	// by devices themselves for methods

	return DEVICE_OK;
}


// ********************************************
// ******* BusyThread implementation **********
// ********************************************
BusyThread::BusyThread(UniHub* p) {
	pHub_ = p;
};

BusyThread::~BusyThread() {

};

int BusyThread::svc(void) 
{
	int ret;
	string temp;
	// serial port answer
	string ans;
	string devicename, command, strerr;
	// any time interval
	MM::MMTime interval;
	// time interval for periodic checking for incoming communication
	MM::MMTime intervalPortCheck(3e5);
	MM::MMTime lastPortCheck = pHub_->GetCurrentMMTime();

	// this loop will stop when the hub device is unloaded
	while (!pHub_->stopbusythread_) {
		ans.clear();
		Sleep(500);
		// populate the list of busy devices
		vector<MM::Device*> listOfBusyDevices;
		for (int ii = 0; ii < deviceDescriptionList.size(); ii++) {
			MM::Device* pDev = pHub_->GetDevice(deviceDescriptionList.at(ii).name.c_str());
			if (pDev != 0) if (pDev->Busy()) listOfBusyDevices.push_back(pDev);
		}

		if (listOfBusyDevices.size() > 0 || pHub_->GetCurrentMMTime() - lastPortCheck > intervalPortCheck) {
			lastPortCheck = pHub_->GetCurrentMMTime();
			// check for response
			ret = pHub_->ReceiveAnswer(temp);
			ans.append(temp);
			if (ret != DEVICE_OK) {
				// no communication received therefore check timeout values
				for (size_t ii = 0; ii < listOfBusyDevices.size(); ii++) {
					char cname[MM::MaxStrLength];
					listOfBusyDevices.at(ii)->GetName(cname);
					// check timeout of device cname
					if ((pHub_->GetCurrentMMTime() - pHub_->GetLastCommandTime(string(cname))) > pHub_->GetTimeout(string(cname))) {
						// if this device has timed out
						pHub_->SetBusy(string(cname), false);
						pHub_->ReportTimeoutError(string(cname));
						break;
					}
				}
			}
			else
			{
				// communication received from the serial device
				vector<string> vs = SplitStringIntoWords(ans, ummhwords::sepIn);
				// check the command
				ret = pHub_->CheckIncomingCommand(vs);
				if (ret != DEVICE_OK) {
					pHub_->WriteError(ans, ret);
					ans.clear();
					continue;
				}
				// interpret the command
				devicename = vs[0];
				command = vs[1];
				vector<string> vals = SplitStringIntoWords(vs[2], ummhwords::sepWithin);
				strerr = vals[0];
				vals.erase(vals.begin());
				int err = stoi(strerr);

				// handle errors
				if (err == ummherrors::ctr_ok) {
					// no error, not busy
					pHub_->SetBusy(devicename, false);
				}
				else if (err == ummherrors::ctr_busy) {
					// no error but busy
					pHub_->SetBusy(devicename, true);
					pHub_->SetLastCommandTime(devicename, pHub_->GetCurrentMMTime());
				}
				else {
					// report error
					pHub_->SetBusy(devicename, false);
					pHub_->WriteError(ans, err);
				}
				// report values back to the device
				if (err == ummherrors::ctr_ok || err == ummherrors::ctr_busy) {
					if (vals.size() > 0) ret = pHub_->ReportToDevice(devicename, command, vals);
				}
			}
		}
	}
	return DEVICE_OK;
}

// ********************************************
// ******* UmmhShutter implementation **********
// ********************************************

UmmhShutter::UmmhShutter(const char* name) :
	initialized_(false),
	open_(false)
{
	name_.append(name);
	// parent ID display
	CreateHubIDProperty();
	pHub_ = hub_;
}

UmmhShutter::~UmmhShutter()
{
	Shutdown();
}

void UmmhShutter::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


int UmmhShutter::Initialize()
{
	pHub_ = static_cast<UniHub*>(GetParentHub());
	if (pHub_)
	{
		char hubLabel[MM::MaxStrLength];
		pHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		return DEVICE_COMM_HUB_MISSING;

	if (initialized_) return DEVICE_OK;

	int ret;
	// find this device in the list
	int index = pHub_->GetDeviceIndexFromName(name_);
	vector< mmpropertydescription> pdList = deviceDescriptionList.at(index).properties;
	// set timeout
	SetTimeout(deviceDescriptionList.at(index).timeout);
	// create properties
	for (int ii = 0; ii < pdList.size(); ii++) {
		mmpropertydescription pd = pdList.at(ii);
		if (pd.isPreini) {
			// ignore preinitialization properties
			//mmpropertydescription pdnew = pd;
			//pdnew.name.append(ummhwords::preini_append);
			//pdnew.isReadOnly = true;
			//pdnew.isPreini = false;
			//pdList.push_back(pdnew);
		}
		else { // create post-initialization properties
			ret = CreatePropertyBasedOnDescription(pd);
			if (ret != DEVICE_OK) return ret;
		}
	}

	// assume that the device is not initially busy
	SetBusy(false);

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int UmmhShutter::CreatePropertyBasedOnDescription(mmpropertydescription pd) {
	int ret;
	CPropertyAction* pAct; 
	if (pd.isAction) {
		pAct = new CPropertyAction(this, &UmmhShutter::OnAction);
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly,
				pAct, pd.isPreini);
			SetAllowedValues(pd.name.c_str(), pd.allowedValues);
			if (DEVICE_OK != ret) return ret;
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	else {
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetAllowedValues(pd.name.c_str(), pd.allowedValues);
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	return DEVICE_OK;

}

bool UmmhShutter::Busy()
{
	if (!initialized_) return false;
	return GetBusy();
}

int UmmhShutter::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

int UmmhShutter::SetOpen(bool open)
{
	if (this->IsUpdating()) {
		if (open != 0 && open != 1) return ummherrors::adp_device_command_value_not_allowed;
		open_ = open;
		this->SetUpdating(false);
		return DEVICE_OK;
	}
	
	vector<string> vals;
	vals.push_back(to_string((long long)open));
	string cmd = pHub_->ConvertMethodToCommand(name_,ummhwords::set_open);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	open_ = open;
	SetBusy(true);
	return ret;
}

int UmmhShutter::GetOpen(bool& open)
{
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::get_open);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	if (strcmp(cmd.c_str(),ummhwords::cashed)==0) { // use cashed value
		open = open_;
		return DEVICE_OK;
	}

	vector<string> vals;
	vals.push_back(to_string((long long)open_));	
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	open = open_;
	return ret;
}

int UmmhShutter::Fire(double deltaT)
{
	vector<string> vals;
	vals.push_back(to_string((long double)deltaT));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::fire);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int UmmhShutter::OnAction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int index = pHub_->GetDeviceIndexFromName(name_);
	mmdevicedescription d = deviceDescriptionList.at(index);

	// find out which property is calling
	mmpropertydescription pd;
	size_t propertyIndex_ = (size_t)-1;
	for (int ii = 0; ii < d.properties.size(); ii++) {
		pd = d.properties.at(ii);
		if (pd.name.compare(pProp->GetName()) == 0) {
			propertyIndex_ = ii;
			break;
		}
	}

	int ret;
	stringstream ss;
	string ans;
	string s;
	long vlong;
	double vdouble;

	if (pd.type == MM::PropertyType::String) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set(pd.valueString.c_str());
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(s);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueString = s;
			ss << s << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	} 
	else if (pd.type == MM::PropertyType::Integer) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((long)pd.valueInteger);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vlong);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueInteger = (int)vlong;
			ss << (int)vlong << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	} 
	else if (pd.type == MM::PropertyType::Float) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)pd.valueFloat);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vdouble);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueFloat = (float)vdouble;
			ss << (float)vdouble << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}

	return DEVICE_OK;

}


// ********************************************
// **** UmmhStateDevice implementation *********
// ********************************************

UmmhStateDevice::UmmhStateDevice(const char* name) :
	initialized_(false),
	numberOfPositions_(0),
	positionAkaState_(0)
{
	name_.append(name);
	// parent ID display
	CreateHubIDProperty();
	pHub_ = hub_;
}

UmmhStateDevice::~UmmhStateDevice()
{
	Shutdown();
}

void UmmhStateDevice::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


int UmmhStateDevice::Initialize()
{
	pHub_ = static_cast<UniHub*>(GetParentHub());
	if (pHub_)
	{
		char hubLabel[MM::MaxStrLength];
		pHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		return DEVICE_COMM_HUB_MISSING;

	if (initialized_) return DEVICE_OK;

	int ret;
	// find this device in the list
	int index = pHub_->GetDeviceIndexFromName(name_);
	vector< mmpropertydescription> pdList = deviceDescriptionList.at(index).properties;
	// set timeout
	SetTimeout(deviceDescriptionList.at(index).timeout);
	// create properties
	for (int ii = 0; ii < pdList.size(); ii++) {
		mmpropertydescription pd = pdList.at(ii);
		if (pd.isPreini) {
			// ignore preinitialization properties
			//mmpropertydescription pdnew = pd;
			//pdnew.name.append(ummhwords::preini_append);
			//pdnew.isReadOnly = true;
			//pdnew.isPreini = false;
			//pdList.push_back(pdnew);
		}
		else { // create post-initialization properties
			// special properties dealt with by MM core
			if (strcmp(pd.name.c_str(), MM::g_Keyword_Label) == 0) {
				CPropertyAction* pAct = new CPropertyAction(this, &CStateBase::OnLabel);
				ret = CreateStringProperty(MM::g_Keyword_Label, pd.valueString.c_str(), false, pAct);
				for (int jj = 0; jj < pd.allowedValues.size(); jj++) {
					AddAllowedValue(MM::g_Keyword_Label, pd.allowedValues.at(jj).c_str());
				}
				if (ret != DEVICE_OK) return ret;
			}
			else {
				ret = CreatePropertyBasedOnDescription(pd);
				if (ret != DEVICE_OK) return ret;
			}
		}
	}

	if (this->HasProperty(MM::g_Keyword_State)) {
		// determine the number of positions
		double lowerLimit, upperLimit;
		ret = this->GetPropertyLowerLimit(MM::g_Keyword_State, lowerLimit);
		if (ret != DEVICE_OK) return ret;
		ret = this->GetPropertyUpperLimit(MM::g_Keyword_State, upperLimit);
		if (ret != DEVICE_OK) return ret;
		numberOfPositions_ = (unsigned long)(upperLimit - lowerLimit + 1);
		unsigned long posFirst = (unsigned long)lowerLimit;
		unsigned long posLast = (unsigned long)upperLimit;
		if (this->HasProperty(MM::g_Keyword_Label)) {
			// assign labels
			mmpropertydescription pd;
			for (int ii = 0; ii < pdList.size(); ii++) {
				pd = pdList.at(ii);
				if (strcmp(pd.name.c_str(), MM::g_Keyword_Label) == 0) {
					for (unsigned long jj = posFirst; jj <= posLast; jj++) {
						SetPositionLabel(jj, pd.allowedValues.at(jj-posFirst).c_str());
					}
				}
			}
		}
	}

	// assume that the device is not initially busy
	SetBusy(false);

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int UmmhStateDevice::CreatePropertyBasedOnDescription(mmpropertydescription pd) {
	int ret;
	CPropertyAction* pAct;
	if (pd.isAction) {
		pAct = new CPropertyAction(this, &UmmhStateDevice::OnAction);
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly,
				pAct, pd.isPreini);
			SetAllowedValues(pd.name.c_str(), pd.allowedValues);
			if (DEVICE_OK != ret) return ret;
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	else {
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetAllowedValues(pd.name.c_str(), pd.allowedValues);
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	return DEVICE_OK;

}

bool UmmhStateDevice::Busy()
{
	if (!initialized_) return false;
	return GetBusy();
}

int UmmhStateDevice::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

unsigned long UmmhStateDevice::GetNumberOfPositions() const
{
	return numberOfPositions_;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int UmmhStateDevice::OnAction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int index = pHub_->GetDeviceIndexFromName(name_);
	mmdevicedescription d = deviceDescriptionList.at(index);

	// find out which property is calling
	mmpropertydescription pd;
	size_t propertyIndex_ = (size_t)-1;
	for (int ii = 0; ii < d.properties.size(); ii++) {
		pd = d.properties.at(ii);
		if (pd.name.compare(pProp->GetName()) == 0) {
			propertyIndex_ = ii;
			break;
		}
	}

	int ret;
	stringstream ss;
	string ans;
	string s;
	long vlong;
	double vdouble;

	if (pd.type == MM::PropertyType::String) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set(pd.valueString.c_str());
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(s);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueString = s;
			ss << s << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Integer) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((long)pd.valueInteger);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vlong);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueInteger = (int)vlong;
			ss << (int)vlong << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Float) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)pd.valueFloat);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vdouble);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueFloat = (float)vdouble;
			ss << (float)vdouble << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}

	return DEVICE_OK;

}

// ********************************************
// ********* UmmhStage implementation **********
// ********************************************

UmmhStage::UmmhStage(const char* name) :
	initialized_(false),
	position_um_(0),
	stepSize_um_(1.0)
{
	name_.append(name);
	// parent ID display
	CreateHubIDProperty();
	pHub_ = hub_;
}

UmmhStage::~UmmhStage()
{
	Shutdown();
}

void UmmhStage::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


int UmmhStage::Initialize()
{
	pHub_ = static_cast<UniHub*>(GetParentHub());
	if (pHub_)
	{
		char hubLabel[MM::MaxStrLength];
		pHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		return DEVICE_COMM_HUB_MISSING;

	if (initialized_) return DEVICE_OK;

	int ret;
	// find this device in the list
	int index = pHub_->GetDeviceIndexFromName(name_);
	vector< mmpropertydescription> pdList = deviceDescriptionList.at(index).properties;
	// set timeout
	SetTimeout(deviceDescriptionList.at(index).timeout);
	// create properties
	for (int ii = 0; ii < pdList.size(); ii++) {
		mmpropertydescription pd = pdList.at(ii);
		if (pd.isPreini) {
			// ignore preinitialization properties
			//mmpropertydescription pdnew = pd;
			//pdnew.name.append(ummhwords::preini_append);
			//pdnew.isReadOnly = true;
			//pdnew.isPreini = false;
			//pdList.push_back(pdnew);
		}
		else { // create post-initialization properties
			ret = CreatePropertyBasedOnDescription(pd);
			if (ret != DEVICE_OK) return ret;
		}
	}

	// get limits from Position property
	if (this->HasProperty(MM::g_Keyword_Position)) {
		for (int ii = 0; ii < pdList.size(); ii++) {
			mmpropertydescription pd = pdList.at(ii);
			if (strcmp(pd.name.c_str(), MM::g_Keyword_Position) == 0) {
				if (pd.type == MM::PropertyType::Integer) {
					lowerLimit_um_ = pd.lowerLimitInteger;
					upperLimit_um_ = pd.upperLimitInteger;
				}
				else {
					lowerLimit_um_ = pd.lowerLimitFloat;
					upperLimit_um_ = pd.upperLimitFloat;
				}
				break;
			}
		}

	}

	// assume that the device is not initially busy
	SetBusy(false);

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int UmmhStage::CreatePropertyBasedOnDescription(mmpropertydescription pd) {
	int ret;
	CPropertyAction* pAct;
	if (pd.isAction) {
		pAct = new CPropertyAction(this, &UmmhStage::OnAction);
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly,
				pAct, pd.isPreini);
			SetAllowedValues(pd.name.c_str(), pd.allowedValues);
			if (DEVICE_OK != ret) return ret;
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	else {
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetAllowedValues(pd.name.c_str(), pd.allowedValues);
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	return DEVICE_OK;

}

bool UmmhStage::Busy()
{
	if (!initialized_) return false;
	return GetBusy();
}

int UmmhStage::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

int UmmhStage::SetPositionUm(double pos)
{
	if (this->IsUpdating()) {
		if (pos < lowerLimit_um_ || pos > upperLimit_um_) return ummherrors::adp_device_command_value_not_allowed;
		position_um_ = pos;
		// update Position property if available
		if (this->HasProperty(MM::g_Keyword_Position)) {
			int index = pHub_->GetDeviceIndexFromName(name_);
			mmdevicedescription d = deviceDescriptionList.at(index);
			mmpropertydescription pd;
			size_t propertyIndex_ = (size_t)-1;
			for (int ii = 0; ii < d.properties.size(); ii++) {
				pd = d.properties.at(ii);
				if (pd.name.compare(MM::g_Keyword_Position) == 0) {
					propertyIndex_ = ii;
					break;
				}
			}
			deviceDescriptionList.at(index).properties.at(propertyIndex_).valueFloat = (float)position_um_;
			OnPropertyChanged(pd.name.c_str(), to_string((long double)position_um_).c_str());
		}
		this->SetUpdating(false);
		return DEVICE_OK;
	}

	vector<string> vals;
	vals.push_back(to_string((long long)pos));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::set_position_um);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	position_um_ = pos;
	SetBusy(true);
	return ret;
}

int UmmhStage::GetPositionUm(double& pos)
{
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::get_position_um);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	if (strcmp(cmd.c_str(), ummhwords::cashed) == 0) { // use cashed value
		pos = position_um_;
		return DEVICE_OK;
	}

	vector<string> vals;
	vals.push_back(to_string((long double)position_um_));
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	pos = position_um_;
	return ret;
}

int UmmhStage::Home()
{
	vector<string> vals;
	vals.push_back(to_string((long long)0));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::home);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	return ret;
}

int UmmhStage::Stop()
{
	vector<string> vals;
	vals.push_back(to_string((long long)0));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::stop);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int UmmhStage::OnAction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int index = pHub_->GetDeviceIndexFromName(name_);
	mmdevicedescription d = deviceDescriptionList.at(index);

	// find out which property is calling
	mmpropertydescription pd;
	size_t propertyIndex_ = (size_t)-1;
	for (int ii = 0; ii < d.properties.size(); ii++) {
		pd = d.properties.at(ii);
		if (pd.name.compare(pProp->GetName()) == 0) {
			propertyIndex_ = ii;
			break;
		}
	}

	int ret;
	stringstream ss;
	string ans;
	string s;
	long vlong;
	double vdouble;

	if (pd.type == MM::PropertyType::String) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set(pd.valueString.c_str());
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(s);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueString = s;
			ss << s << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Integer) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((long)pd.valueInteger);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vlong);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueInteger = (int)vlong;
			ss << (int)vlong << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Float) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)pd.valueFloat);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vdouble);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueFloat = (float)vdouble;
			ss << (float)vdouble << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}

	return DEVICE_OK;

}

// ********************************************
// ********* UmmhStage implementation **********
// ********************************************

UmmhXYStage::UmmhXYStage(const char* name) :
	initialized_(false),
	positionX_um_(0),
	stepSizeX_um_(1.0),
	positionY_um_(0),
	stepSizeY_um_(1.0)
{
	name_.append(name);
	// parent ID display
	CreateHubIDProperty();
	pHub_ = hub_;
}

UmmhXYStage::~UmmhXYStage()
{
	Shutdown();
}

void UmmhXYStage::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


int UmmhXYStage::Initialize()
{
	pHub_ = static_cast<UniHub*>(GetParentHub());
	if (pHub_)
	{
		char hubLabel[MM::MaxStrLength];
		pHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		return DEVICE_COMM_HUB_MISSING;

	if (initialized_) return DEVICE_OK;

	int ret;
	// find this device in the list
	int index = pHub_->GetDeviceIndexFromName(name_);
	vector< mmpropertydescription> pdList = deviceDescriptionList.at(index).properties;
	// set timeout
	SetTimeout(deviceDescriptionList.at(index).timeout);
	// create properties
	for (int ii = 0; ii < pdList.size(); ii++) {
		mmpropertydescription pd = pdList.at(ii);
		if (pd.isPreini) {
			// ignore preinitialization properties
			//mmpropertydescription pdnew = pd;
			//pdnew.name.append(ummhwords::preini_append);
			//pdnew.isReadOnly = true;
			//pdnew.isPreini = false;
			//pdList.push_back(pdnew);
		}
		else { // create post-initialization properties
			ret = CreatePropertyBasedOnDescription(pd);
			if (ret != DEVICE_OK) return ret;
		}
	}

	// get limits from Position property
	if (this->HasProperty(ummhwords::position_x)) {
		for (int ii = 0; ii < pdList.size(); ii++) {
			mmpropertydescription pd = pdList.at(ii);
			if (strcmp(pd.name.c_str(), ummhwords::position_x) == 0) {
				if (pd.type == MM::PropertyType::Integer) {
					lowerLimitX_um_ = pd.lowerLimitInteger;
					upperLimitX_um_ = pd.upperLimitInteger;
				}
				else {
					lowerLimitX_um_ = pd.lowerLimitFloat;
					upperLimitX_um_ = pd.upperLimitFloat;
				}
				break;
			}
		}

	}
	if (this->HasProperty(ummhwords::position_y)) {
		for (int ii = 0; ii < pdList.size(); ii++) {
			mmpropertydescription pd = pdList.at(ii);
			if (strcmp(pd.name.c_str(), ummhwords::position_y) == 0) {
				if (pd.type == MM::PropertyType::Integer) {
					lowerLimitY_um_ = pd.lowerLimitInteger;
					upperLimitY_um_ = pd.upperLimitInteger;
				}
				else {
					lowerLimitY_um_ = pd.lowerLimitFloat;
					upperLimitY_um_ = pd.upperLimitFloat;
				}
				break;
			}
		}

	}

	// assume that the device is not initially busy
	SetBusy(false);

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int UmmhXYStage::CreatePropertyBasedOnDescription(mmpropertydescription pd) {
	int ret;
	CPropertyAction* pAct;
	if (pd.isAction) {
		pAct = new CPropertyAction(this, &UmmhXYStage::OnAction);
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly,
				pAct, pd.isPreini);
			SetAllowedValues(pd.name.c_str(), pd.allowedValues);
			if (DEVICE_OK != ret) return ret;
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	else {
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetAllowedValues(pd.name.c_str(), pd.allowedValues);
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	return DEVICE_OK;

}

bool UmmhXYStage::Busy()
{
	if (!initialized_) return false;
	return GetBusy();
}

int UmmhXYStage::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

int UmmhXYStage::SetPositionUm(double posX, double posY)
{
	if (this->IsUpdating()) {
		if (posX < lowerLimitX_um_ || posX > upperLimitX_um_ ||
			posY < lowerLimitY_um_ || posY > upperLimitY_um_) {
			return ummherrors::adp_device_command_value_not_allowed;
		}
		positionX_um_ = posX;
		positionY_um_ = posY;
		// update Position properties if available
		int index = pHub_->GetDeviceIndexFromName(name_);
		mmdevicedescription d = deviceDescriptionList.at(index);
		mmpropertydescription pd;
		size_t propertyIndex_ = (size_t)-1;
		if (this->HasProperty(ummhwords::position_x)) {
			for (int ii = 0; ii < d.properties.size(); ii++) {
				pd = d.properties.at(ii);
				if (pd.name.compare(ummhwords::position_x) == 0) {
					propertyIndex_ = ii;
					break;
				}
			}
			deviceDescriptionList.at(index).properties.at(propertyIndex_).valueFloat = (float)positionX_um_;
			OnPropertyChanged(pd.name.c_str(), to_string((long double)positionX_um_).c_str());
		}
		if (this->HasProperty(ummhwords::position_y)) {
			for (int ii = 0; ii < d.properties.size(); ii++) {
				pd = d.properties.at(ii);
				if (pd.name.compare(ummhwords::position_y) == 0) {
					propertyIndex_ = ii;
					break;
				}
			}
			deviceDescriptionList.at(index).properties.at(propertyIndex_).valueFloat = (float)positionY_um_;
			OnPropertyChanged(pd.name.c_str(), to_string((long double)positionY_um_).c_str());
		}
		this->SetUpdating(false);
		return DEVICE_OK;
	}

	vector<string> vals;
	vals.push_back(to_string((long double)posX));
	vals.push_back(to_string((long double)posY));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::set_position_um);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	positionX_um_ = posX;
	positionY_um_ = posY;
	SetBusy(true);
	return ret;
}

int UmmhXYStage::GetPositionUm(double& posX, double& posY)
{
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::get_position_um);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	if (strcmp(cmd.c_str(), ummhwords::cashed) == 0) { // use cashed value
		posX = positionX_um_;
		posY = positionY_um_;
		return DEVICE_OK;
	}

	vector<string> vals;
	vals.push_back(to_string((long double)positionX_um_));
	vals.push_back(to_string((long double)positionY_um_));
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	posX = positionX_um_;
	posY = positionY_um_;
	return ret;
}

int UmmhXYStage::Home()
{
	vector<string> vals;
	vals.push_back(to_string((long long)0));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::home);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	return ret;
}

int UmmhXYStage::Stop()
{
	vector<string> vals;
	vals.push_back(to_string((long long)0));
	string cmd = pHub_->ConvertMethodToCommand(name_, ummhwords::stop);
	if (cmd.length() == 0) return DEVICE_ERR;
	if (strcmp(cmd.c_str(), ummhwords::not_supported) == 0) return DEVICE_UNSUPPORTED_COMMAND;
	SetLastCommandTime(GetCurrentMMTime());
	int ret = pHub_->MakeAndSendOutputCommand(name_, cmd, vals);
	SetBusy(true);
	return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int UmmhXYStage::OnAction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int index = pHub_->GetDeviceIndexFromName(name_);
	mmdevicedescription d = deviceDescriptionList.at(index);

	// find out which property is calling
	mmpropertydescription pd;
	size_t propertyIndex_ = (size_t)-1;
	for (int ii = 0; ii < d.properties.size(); ii++) {
		pd = d.properties.at(ii);
		if (pd.name.compare(pProp->GetName()) == 0) {
			propertyIndex_ = ii;
			break;
		}
	}

	int ret;
	stringstream ss;
	string ans;
	string s;
	long vlong;
	double vdouble;

	if (pd.type == MM::PropertyType::String) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set(pd.valueString.c_str());
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(s);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueString = s;
			ss << s << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Integer) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((long)pd.valueInteger);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vlong);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			if (strcmp(pProp->GetName().c_str(),ummhwords::position_x) == 0) {
				ss << (int)vlong << ummhwords::sepWithin << (int)positionY_um_ << ummhwords::sepEnd;
			}
			else if (strcmp(pProp->GetName().c_str(), ummhwords::position_y) == 0) {
				ss << (int)positionX_um_ << ummhwords::sepWithin << (int)vlong << ummhwords::sepEnd;
			}
			else {
				ss << (int)vlong << ummhwords::sepEnd;
			}
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Float) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)pd.valueFloat);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vdouble);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			if (strcmp(pProp->GetName().c_str(), ummhwords::position_x) == 0) {
				ss << (float)vdouble << ummhwords::sepWithin << (float)positionY_um_ << ummhwords::sepEnd;
			}
			else if (strcmp(pProp->GetName().c_str(), ummhwords::position_y) == 0) {
				ss << (float)positionX_um_ << ummhwords::sepWithin << (float)vdouble << ummhwords::sepEnd;
			}
			else {
				ss << (float)vdouble << ummhwords::sepEnd;
			}
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}

	return DEVICE_OK;

}


// ********************************************
// ******* UmmhGeneric implementation **********
// ********************************************

UmmhGeneric::UmmhGeneric(const char* name) :
	initialized_(false)
{
	name_.append(name);
	// parent ID display
	CreateHubIDProperty();
	pHub_ = hub_;
}

UmmhGeneric::~UmmhGeneric()
{
	Shutdown();
}

void UmmhGeneric::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


int UmmhGeneric::Initialize()
{
	pHub_ = static_cast<UniHub*>(GetParentHub());
	if (pHub_)
	{
		char hubLabel[MM::MaxStrLength];
		pHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		return DEVICE_COMM_HUB_MISSING;

	if (initialized_) return DEVICE_OK;

	int ret;
	// find this device in the list
	int index = pHub_->GetDeviceIndexFromName(name_);
	vector< mmpropertydescription> pdList = deviceDescriptionList.at(index).properties;
	// set timeout
	SetTimeout(deviceDescriptionList.at(index).timeout);
	// create properties
	for (int ii = 0; ii < pdList.size(); ii++) {
		mmpropertydescription pd = pdList.at(ii);
		if (pd.isPreini) {
			// ignore preinitialization properties
			//mmpropertydescription pdnew = pd;
			//pdnew.name.append(ummhwords::preini_append);
			//pdnew.isReadOnly = true;
			//pdnew.isPreini = false;
			//pdList.push_back(pdnew);
		}
		else { // create post-initialization properties
			ret = CreatePropertyBasedOnDescription(pd);
			if (ret != DEVICE_OK) return ret;
		}
	}

	// assume that the device is not initially busy
	SetBusy(false);

	ret = UpdateStatus();
	if (ret != DEVICE_OK) return ret;

	initialized_ = true;

	return DEVICE_OK;
}

int UmmhGeneric::CreatePropertyBasedOnDescription(mmpropertydescription pd) {
	int ret;
	CPropertyAction* pAct;
	if (pd.isAction) {
		pAct = new CPropertyAction(this, &UmmhGeneric::OnAction);
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly,
				pAct, pd.isPreini);
			SetAllowedValues(pd.name.c_str(), pd.allowedValues);
			if (DEVICE_OK != ret) return ret;
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly,
				pAct, pd.isPreini);
			if (DEVICE_OK != ret) return ret;
			SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	else {
		if (pd.type == MM::PropertyType::String) {
			ret = CreateStringProperty(pd.name.c_str(), pd.valueString.c_str(), pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetAllowedValues(pd.name.c_str(), pd.allowedValues);
		}
		else if (pd.type == MM::PropertyType::Integer) {
			ret = CreateIntegerProperty(pd.name.c_str(), pd.valueInteger, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitInteger, pd.upperLimitInteger);
		}
		else if (pd.type == MM::PropertyType::Float) {
			ret = CreateFloatProperty(pd.name.c_str(), pd.valueFloat, pd.isReadOnly);
			if (DEVICE_OK != ret) return ret;
			if (!pd.isReadOnly) SetPropertyLimits(pd.name.c_str(), pd.lowerLimitFloat, pd.upperLimitFloat);
		}
	}
	return DEVICE_OK;

}

bool UmmhGeneric::Busy()
{
	if (!initialized_) return false;
	return GetBusy();
}

int UmmhGeneric::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int UmmhGeneric::OnAction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int index = pHub_->GetDeviceIndexFromName(name_);
	mmdevicedescription d = deviceDescriptionList.at(index);

	// find out which property is calling
	mmpropertydescription pd;
	size_t propertyIndex_ = (size_t)-1;
	for (int ii = 0; ii < d.properties.size(); ii++) {
		pd = d.properties.at(ii);
		if (pd.name.compare(pProp->GetName()) == 0) {
			propertyIndex_ = ii;
			break;
		}
	}

	int ret;
	stringstream ss;
	string ans;
	string s;
	long vlong;
	double vdouble;

	if (pd.type == MM::PropertyType::String) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set(pd.valueString.c_str());
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(s);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueString = s;
			ss << s << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Integer) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((long)pd.valueInteger);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vlong);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueInteger = (int)vlong;
			ss << (int)vlong << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}
	else if (pd.type == MM::PropertyType::Float) {
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)pd.valueFloat);
		}
		else if (eAct == MM::AfterSet)
		{
			// send the command out
			pProp->Get(vdouble);
			ss << d.name << ummhwords::sepOut << pd.cmdAction << ummhwords::sepOut;
			//deviceDescriptionList.at(index).deviceProperties.at(propertyIndex_).propertyValueFloat = (float)vdouble;
			ss << (float)vdouble << ummhwords::sepEnd;
			if (pd.isPreini) { // get the answer here, don't set busy status
				pHub_->SendCommand(ss.str());
				ret = pHub_->ReceiveAndWaitForAnswer(ans, MM::MMTime(1e6));
			}
			else { // set busy status, send command and exit
				SetLastCommandTime(GetCurrentMMTime());
				pHub_->SendCommand(ss.str());
				SetBusy(true);
			}
		}
	}

	return DEVICE_OK;

}

