///////////////////////////////////////////////////////////////////////////////
// FILE:          ScientificaMotion8.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Scientifica Motion 8 rack adapter
// COPYRIGHT:     University of California, San Francisco, 2006
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
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/01/2006
//
//				  Scientifica Specific Parts
// AUTHOR:		  Matthew Player (ElecSoft Solutions)

#include "ScientificaMotion8.h"
#include "ScientificaRxPacket.h"

#include "ModuleInterface.h"
#include <sstream>
#include <cstdio>
#include <iomanip>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2
#define AXIS_F 6

const char* g_DeviceNameM8Hub = "Scientifica-Motion8-Hub";
const char* g_DeviceNameM8XY_Device1 = "Scientifica-Motion8-XY_Device_1";
const char* g_DeviceNameM8Z_Device1 = "Scientifica-Motion8-Z_Device_1";
const char* g_DeviceNameM8XY_Device2 = "Scientifica-Moiton8-XY_Device_2";
const char* g_DeviceNameM8Z_Device2 = "Scientifica-Moition8-Z_Device_2";
const char* g_DeviceNameFilter_Device1 = "Scientifica-Motion8-Filter_Device_1";
const char* g_DeviceNameFilter_Device2 = "Scientifica-Motion8-Filter_Device_2";

// static lock
MMThreadLock ScientificaMotion8Hub::lock_;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_DeviceNameM8Hub, MM::HubDevice, "Hub (required)");
    RegisterDevice(g_DeviceNameM8XY_Device1, MM::XYStageDevice, "XY Stage (Device 1)");
    RegisterDevice(g_DeviceNameM8Z_Device1, MM::StageDevice, "Z Stage (Device 1)");
    RegisterDevice(g_DeviceNameM8XY_Device2, MM::XYStageDevice, "XY Stage (Device 2)");
    RegisterDevice(g_DeviceNameM8Z_Device2, MM::StageDevice, "Z Stage (Device 2)");
    RegisterDevice(g_DeviceNameFilter_Device1, MM::StateDevice, "Filter Wheel (Device 1)");
    RegisterDevice(g_DeviceNameFilter_Device2, MM::StateDevice, "Filter Wheel (Device 2)");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;
    
    if (strcmp(deviceName, g_DeviceNameM8Hub) == 0)
    {
        return new ScientificaMotion8Hub;
    }
    else if (strcmp(deviceName, g_DeviceNameM8XY_Device1) == 0)
	{
		return new M8XYStage(0);
	}
	else if (strcmp(deviceName, g_DeviceNameM8Z_Device1) == 0)
	{
		return new M8ZStage(0);
	}
	else if (strcmp(deviceName, g_DeviceNameM8XY_Device2) == 0)
	{
		return new M8XYStage(1);
	}
	else if (strcmp(deviceName, g_DeviceNameM8Z_Device2) == 0)
	{
		return new M8ZStage(1);
	}
    else if (strcmp(deviceName, g_DeviceNameFilter_Device1) == 0)
    {
        return new M8FilterCubeTurret(0);
    }
    else if (strcmp(deviceName, g_DeviceNameFilter_Device2) == 0)
	{
		return new M8FilterCubeTurret(1);
	}

    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// M8XYStage implementation
///////////////////////////////////////////////////////////////////////////////

ScientificaMotion8Hub::ScientificaMotion8Hub() :
    initialized_ (false)
{
	InitializeDefaultErrorMessages();

	// Parent ID display
	CreateHubIDProperty();

    CPropertyAction* pAct = new CPropertyAction(this, &ScientificaMotion8Hub::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

    device_1_x_channel_ = 0xFF;
    device_1_y_channel_ = 0xFF;
    device_1_z_channel_ = 0xFF;
    device_1_f_channel_ = 0xFF;

    device_2_x_channel_ = 0xFF;
    device_2_y_channel_ = 0xFF;
    device_2_z_channel_ = 0xFF;
    device_2_f_channel_ = 0xFF;
}

ScientificaMotion8Hub::~ScientificaMotion8Hub()
{
    Shutdown();
}

void ScientificaMotion8Hub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameM8Hub);
}

bool ScientificaMotion8Hub::Busy()
{
    return false;
}

//Is DetectDevice() implemented
bool ScientificaMotion8Hub::SupportsDeviceDetection(void)
{
    return true;
}

//Used to automate discovery of correct serial port
MM::DeviceDetectionStatus ScientificaMotion8Hub::DetectDevice(void)
{
    if (initialized_)
        return MM::CanCommunicate;

    MM::DeviceDetectionStatus result = MM::Misconfigured;
    char answerTO[MM::MaxStrLength];

    try
    {
        std::string portLowerCase = port_;
        for (std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
        {
            *its = (char)tolower(*its);
        }
        if (0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown"))
        {
            result = MM::CanNotCommunicate;

            GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

            GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115200");
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");

            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "300.0");
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");

            MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
            pS->Initialize();

            CDeviceUtils::SleepMs(1000);
            MMThreadGuard myLock(lock_);
            PurgeComPort(port_.c_str());

            std::string version;
            bool supportedVersion = CheckControllerVersion();
            if (!supportedVersion)
            {
                LogMessage("Controller needs updating to 0.9.27 or above. Please use LinLab 3 0.5.16 or above to update the controller.");
                result = MM::Misconfigured;
            }
            else
            {
                int ret = ReadControllerMap();
                if (ret == DEVICE_OK &&
                    ((device_1_x_channel_ != 0xFF) || (device_1_y_channel_ != 0xFF) || (device_1_z_channel_ != 0xFF) || (device_2_x_channel_ != 0xFF) || (device_2_y_channel_ != 0xFF) || (device_2_z_channel_ != 0xFF)))
                {
                    result = MM::CanCommunicate;
                }
            }

            pS->Shutdown();

            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);
        }
    }
    catch (...)
    {
        LogMessage("Exception in DetectDevice!", false);
	}
    
	return result;
}

ScientificaRxPacket* ScientificaMotion8Hub::WriteRead(ScientificaTxPacket* tx, int expected_length)
{
    ScientificaRxPacket* rx_packet = NULL;

    MMThreadGuard myLock(lock_);

    PurgeComPort(port_.c_str());
    
    const unsigned char* data = tx->GetPacketToSend();
    int len = tx->GetEncodedLength();

    int return_status = WriteToComPort(port_.c_str(), data, len);

    if (return_status != DEVICE_OK)
    {
		return NULL;
	}

    unsigned char rxBuffer[256]; //256 is the maximum size of a packet
    int bufferIndex = 0;
    unsigned long actualBytesRead = 0;

    MM::MMTime startTime = GetCurrentMMTime();
    bool readZero = false;

    while (!readZero && ((GetCurrentMMTime() - startTime).getMsec() < 250))
    {
        return_status = ReadFromComPort(port_.c_str(), &rxBuffer[bufferIndex], 256 - bufferIndex, actualBytesRead);
        if (return_status != DEVICE_OK)
            break;

        for (unsigned int i = bufferIndex; i < (bufferIndex + actualBytesRead); i++)
        {
            if (rxBuffer[i] == 0)
            {
                readZero = true;
                break;
            }
        }

        bufferIndex += actualBytesRead;
    }

    if (!readZero)
    {
        rx_packet = NULL;
    }

    if (bufferIndex > 0)
    {
        rx_packet = new ScientificaRxPacket(rxBuffer, bufferIndex - 1); //rxBuffer is decoded into to rx_packet

        if (rx_packet->RemainingBytes() < expected_length)
        {
			delete rx_packet;
            rx_packet = NULL;
		}
    }

    return rx_packet;
}


bool ScientificaMotion8Hub::CheckControllerVersion()
{
    bool supportedVersion = false;
    std::string version;
    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 0, 1);
    ScientificaRxPacket* rxPacket = NULL;

    rxPacket = WriteRead(txPacket, 6);

    if (rxPacket != NULL)
    {
        uint16_t major;
        uint16_t minor;
        uint16_t patch;

        rxPacket->GetUInt16(&major);
        rxPacket->GetUInt16(&minor);
        rxPacket->GetUInt16(&patch);

        if (minor >= 9 || (minor == 9 && patch > 27))
        {
            supportedVersion = true;
        }

        std::ostringstream oss;
        oss << major << "." << minor << "." << patch;
        version = oss.str();

        LogMessage("Controller version: " + version, false);

        delete rxPacket;
    }

    return supportedVersion;
}

int ScientificaMotion8Hub::Stop(uint8_t device)
{
    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 7);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    txPacket->AddUInt8(device);
    rxPacket = WriteRead(txPacket, 0);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        delete rxPacket;
    }

    return ret;
}

int ScientificaMotion8Hub::SetPosition(uint8_t device, uint8_t axis, long steps)
{
    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 0xC);
    txPacket->AddUInt8(device);
    txPacket->AddUInt8(axis);
    txPacket->AddInt32(steps);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    rxPacket = WriteRead(txPacket, 0);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        delete rxPacket;
    }

    return ret;
}

int ScientificaMotion8Hub::IsMoving(uint8_t device, bool* is_moving)
{
    if(is_moving == NULL)
		return DEVICE_ERR;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 0xF);
    txPacket->AddUInt8(device);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;
    rxPacket = WriteRead(txPacket, 1);

    if (rxPacket == NULL)
	{
		ret = DEVICE_SERIAL_TIMEOUT;
	}
	else
	{
		uint8_t moving;
		rxPacket->GetByte(&moving);
		*is_moving = moving != 0;
		delete rxPacket;
	}
    
	return ret;
}

int ScientificaMotion8Hub::ReadControllerMap(void)
{
    unsigned char tx[3];
    tx[0] = 0xBB;
    tx[1] = 0;
    tx[2] = 6;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 0, 6);
    ScientificaRxPacket* rxPacket = NULL;

    device_1_x_channel_ = 0xFF;
    device_1_y_channel_ = 0xFF;
    device_1_z_channel_ = 0xFF;
    device_1_f_channel_ = 0xFF;

    device_2_x_channel_ = 0xFF;
    device_2_y_channel_ = 0xFF;
    device_2_z_channel_ = 0xFF;
    device_2_f_channel_ = 0xFF;


    int ret = DEVICE_OK;
    rxPacket = WriteRead(txPacket, 16);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        for (uint8_t ch = 0; ch < 8; ch++)
        {
            uint8_t device;
            uint8_t axis;

            rxPacket->GetByte(&device);
            rxPacket->GetByte(&axis);

            if (device == 0)
            {
                if (axis == AXIS_X)
                    device_1_x_channel_ = ch;
                else if (axis == AXIS_Y)
                    device_1_y_channel_ = ch;
                else if (axis == AXIS_Z)
                    device_1_z_channel_ = ch;
                else if (axis == AXIS_F)
                    device_1_f_channel_ = ch;

            }
            else if (device == 1)
            {
                if (axis == AXIS_X)
                    device_2_x_channel_ = ch;
                else if (axis == AXIS_Y)
                    device_2_y_channel_ = ch;
                else if (axis == AXIS_Z)
                    device_2_z_channel_ = ch;
                else if (axis == AXIS_F)
                    device_2_f_channel_ = ch;
            }
        }

        delete rxPacket;
    }

    return ret;
}


int ScientificaMotion8Hub::Initialize()
{
    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameM8Hub, MM::String, true);
    if (DEVICE_OK != ret)
        return ret;

    CDeviceUtils::SleepMs(2000);

    MMThreadGuard myLock(lock_);

    initialized_ = true;

    return DEVICE_OK;
}


//Detext and instantiate all avalible chile peripherals
int ScientificaMotion8Hub::DetectInstalledDevices()
{
    if (MM::CanCommunicate == DetectDevice())
    {
        ReadControllerMap();

        std::vector<std::string> peripherals;
        peripherals.clear();

        if((device_1_x_channel_ != 0xFF) || (device_1_y_channel_ != 0xFF))
        {
            peripherals.push_back(g_DeviceNameM8XY_Device1);
		}
        if (device_1_z_channel_ != 0xFF)
        {
            peripherals.push_back(g_DeviceNameM8Z_Device1);
        }
        if (device_1_f_channel_ != 0xFF)
        {
            peripherals.push_back(g_DeviceNameFilter_Device1);
        }
        if ((device_2_x_channel_ != 0xFF) || (device_2_y_channel_ != 0xFF))
        {
            peripherals.push_back(g_DeviceNameM8XY_Device2);
        }
        if (device_2_z_channel_ != 0xFF)
		{
            peripherals.push_back(g_DeviceNameM8Z_Device2);
		}
        if (device_2_f_channel_ != 0xFF)
        {
            peripherals.push_back(g_DeviceNameFilter_Device2);
        }

        for (size_t i = 0; i < peripherals.size(); i++)
        {
            MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
            if (pDev)
            {
                AddInstalledDevice(pDev);
            }
        }

    }

    return DEVICE_OK;
}

int ScientificaMotion8Hub::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

int ScientificaMotion8Hub::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
    if (pAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (pAct == MM::AfterSet)
    {
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

M8XYStage::M8XYStage(uint8_t device)
{
	CreateProperty(MM::g_Keyword_Name, "Scientifica-Motion8-XYStage", MM::String, true);
    device_ = device;

    name_ = device == 0 ? g_DeviceNameM8XY_Device1 : g_DeviceNameM8XY_Device2;
}

M8XYStage::~M8XYStage()
{

}

bool M8XYStage::Busy()
{
    bool is_moving = false;

    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return true;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return true;

    ret = parentHub->IsMoving(device_, &is_moving);

    return is_moving;
}

void M8XYStage::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

int M8XYStage::Initialize()
{
    return DEVICE_OK;
}

int M8XYStage::Shutdown()
{
	return DEVICE_OK;
}

int M8XYStage::SetPositionSteps(long x, long y)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 3);
    txPacket->AddUInt8(device_);
    txPacket->AddInt32(x);
    txPacket->AddInt32(y);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    rxPacket = parentHub->WriteRead(txPacket, 0);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        delete rxPacket;
    }

    return ret;
}

int M8XYStage::GetPositionSteps(long& x, long& y)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 0, 0x14);
    txPacket->AddUInt8(device_);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    rxPacket = parentHub->WriteRead(txPacket, 29);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        uint8_t device;
        rxPacket->GetByte(&device); //Called to skip the device byte
        int32_t x_from_device;
        int32_t y_from_device;
        rxPacket->GetInt32(&x_from_device);
        rxPacket->GetInt32(&y_from_device);
        x = x_from_device;
        y = y_from_device;
        delete rxPacket;
    }

    return ret;
}

int M8XYStage::Home()
{
	return DEVICE_OK;
}

int M8XYStage::Stop()
{
    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ret = parentHub->Stop(device_);

    return ret;
}

int M8XYStage::SetOrigin()
{
    int ret;
    ret = SetXOrigin();
    if (ret == DEVICE_OK)
        ret = SetYOrigin();

    return ret;
}

int M8XYStage::SetXOrigin()
{
    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ret = parentHub->SetPosition(device_, AXIS_X, 0);

    return ret;
}

int M8XYStage::SetYOrigin()
{
    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ret = parentHub->SetPosition(device_, AXIS_Y, 0);
    return ret;
}

int M8XYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) 
{ 
    (void)xMin;
    (void)xMax;
    (void)yMin;
    (void)yMax;

    return DEVICE_UNSUPPORTED_COMMAND;
}

int M8XYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
    (void)xMin;
    (void)xMax;
    (void)yMin;
    (void)yMax;

    return DEVICE_UNSUPPORTED_COMMAND; 
}

M8ZStage::M8ZStage(uint8_t device)
{
	CreateProperty(MM::g_Keyword_Name, "Scientifica-Motion8-ZStage", MM::String, true);
    device_ = device;


    name_ = device == 0 ? g_DeviceNameM8Z_Device1 : g_DeviceNameM8Z_Device2;
}

M8ZStage::~M8ZStage()
{

}

bool M8ZStage::Busy()
{
    bool is_moving = false;

    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return true;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return true;

    ret = parentHub->IsMoving(device_, &is_moving);

    return is_moving;
}

void M8ZStage::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

int M8ZStage::Initialize()
{
	return DEVICE_OK;
}

int M8ZStage::Shutdown()
{
	return DEVICE_OK;
}

int M8ZStage::SetPositionUm(double pos)
{
    long steps = (long)round(pos / 0.01);
	return SetPositionSteps(steps);

}
int M8ZStage::GetPositionUm(double& pos)
{
    long steps;
    int ret = GetPositionSteps(steps);
    if (ret != DEVICE_OK)
        return ret;

    pos = steps * 0.01;

    return DEVICE_OK;
}

int M8ZStage::SetPositionSteps(long z)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 3);
    txPacket->AddUInt8(device_);
    txPacket->AddUInt8(AXIS_Z);
    txPacket->AddInt32(z);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    rxPacket = parentHub->WriteRead(txPacket, 0);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        delete rxPacket;
    }

    return ret;
}

int M8ZStage::GetPositionSteps(long& z)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 0, 0x14);
    txPacket->AddUInt8(device_);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    rxPacket = parentHub->WriteRead(txPacket, 29);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {
        
        uint8_t device;
       
        rxPacket->GetByte(&device); //Called to skip the device byte

        int32_t ignore_from_device;
        rxPacket->GetInt32(&ignore_from_device); //Called to skip the x value
        rxPacket->GetInt32(&ignore_from_device); //Called to skip the y value
        
        int32_t z_from_device;
        rxPacket->GetInt32(&z_from_device);
        z = z_from_device;

        delete rxPacket;
    }


    return ret;
}

int M8ZStage::Home()
{
	return DEVICE_OK;
}

int M8ZStage::Stop()
{
    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ret = parentHub->Stop(device_);

    return ret;
}

int M8ZStage::SetOrigin()
{
    int ret = DEVICE_OK;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ret = parentHub->SetPosition(device_, AXIS_Z, 0);

    return ret;
}

int M8ZStage::GetLimits(double& min, double& max) 
{ 
    (void)min;
    (void)max;
    return DEVICE_UNSUPPORTED_COMMAND; 
}

M8FilterCubeTurret::M8FilterCubeTurret(uint8_t device)
{
   // CreateProperty(MM::g_Keyword_Name, "Scientifica-Motion8-ZStage", MM::String, true);
    device_ = device;
    numPositions_ = 3;

    name_ = device == 0 ? g_DeviceNameFilter_Device1 : g_DeviceNameFilter_Device2;

}

M8FilterCubeTurret::~M8FilterCubeTurret()
{

}

int M8FilterCubeTurret::Initialize()
{
	return DEVICE_OK;
}

int M8FilterCubeTurret::Shutdown()
{
	return DEVICE_OK;
}

void M8FilterCubeTurret::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

bool M8FilterCubeTurret::Busy()
{
    bool is_moving = false;

    MM::Hub* hub = GetParentHub();
    if (!hub)
        return is_moving;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return is_moving;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 0, 0x11);
    ScientificaRxPacket* rxPacket = NULL;

    rxPacket = parentHub->WriteRead(txPacket, 18);

    if (rxPacket != NULL)
    {

        if (device_ == 0)
        {
            rxPacket->Skip(5); //Skip to Device 1 filter state
        }
		else
		{
			rxPacket->Skip(14); //Skip to Device 2 filter state
		}

        uint8_t filterState;
        rxPacket->GetByte(&filterState);

        is_moving = filterState != 0;

        delete rxPacket;
    }

	return is_moving;
}

int M8FilterCubeTurret::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    LogMessage("M8FilterCubeTurret::PositionGetSet\n", true);

    if (eAct == MM::BeforeGet)
    {
        int filterIndex;
        int ret = GetFilter(filterIndex);
        if (ret != DEVICE_OK)
            return ret;
        pProp->Set((long)filterIndex);
    }
    else if (eAct == MM::AfterSet)
    {
        long filterIndex;
        pProp->Get(filterIndex);

        if ((filterIndex > 0) && (filterIndex <= numPositions_))
            return SetFilter(filterIndex);
        else
            return DEVICE_UNKNOWN_POSITION;
    }

    return DEVICE_OK;
}

int M8FilterCubeTurret::SetFilter(int filterIndex)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 0x0E);
    txPacket->AddUInt8(device_);
    txPacket->AddUInt8((uint8_t)filterIndex);

    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;
    rxPacket = parentHub->WriteRead(txPacket, 18);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
	else
	{
		delete rxPacket;
	}

    return ret;
}

int M8FilterCubeTurret::GetFilter(int& filterIndex)
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaMotion8Hub* parentHub = dynamic_cast<ScientificaMotion8Hub*>(hub);
    if (!parentHub)
        return DEVICE_INTERNAL_INCONSISTENCY;

    ScientificaTxPacket* txPacket = new ScientificaTxPacket(0xBB, 2, 0x0E);
    ScientificaRxPacket* rxPacket = NULL;

    int ret = DEVICE_OK;

    txPacket->AddUInt8((uint8_t)filterIndex);
    rxPacket = parentHub->WriteRead(txPacket, 2);

    if (rxPacket == NULL)
    {
        ret = DEVICE_SERIAL_TIMEOUT;
    }
    else
    {

        rxPacket->Skip(1); //Skip device index
        uint8_t filter;
        rxPacket->GetByte(&filter);

        if (filter < numPositions_)
            filter = filter;
        else
            ret = DEVICE_UNKNOWN_POSITION;


        delete rxPacket;
    }

	return ret;
}