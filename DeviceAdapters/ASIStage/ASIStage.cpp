/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIStage.h"
#include "ASICRISP.h"
#include "ASILED.h"
#include "ASIMagnifier.h"
#include "ASIStateDevice.h"
#include "ASITurret.h"
#include "ASIXYStage.h"
#include "ASIZStage.h"
#include "ASITIRF.h"


MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_ZStageDeviceName, MM::StageDevice, "Z Stage");
    RegisterDevice(g_XYStageDeviceName, MM::XYStageDevice, "XY Stage");
    RegisterDevice(g_CRISPDeviceName, MM::AutoFocusDevice, "CRISP");
    RegisterDevice(g_AZ100TurretName, MM::StateDevice, "AZ100 Turret");
    RegisterDevice(g_StateDeviceName, MM::StateDevice, "State Device");
    RegisterDevice(g_MagnifierDeviceName, MM::MagnifierDevice, "Magnifier");
    RegisterDevice(g_LEDDeviceName, MM::ShutterDevice, "LED");
    RegisterDevice(g_TIRFDeviceName, MM::GenericDevice, "TIRF");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == nullptr)
    {
        return nullptr;
    }

    const std::string name = deviceName;

    if (name == g_ZStageDeviceName)
    {
        return new ZStage();
    }
    else if (name == g_XYStageDeviceName)
    {
        return new XYStage();
    }
    else if (name == g_CRISPDeviceName)
    {
        return new CRISP();
    }
    else if (name == g_AZ100TurretName)
    {
        return new AZ100Turret();
    }
    else if (name == g_StateDeviceName)
    {
        return new StateDevice();
    }
    else if (name == g_MagnifierDeviceName)
    {
        return new Magnifier();
    }
    else if (name == g_LEDDeviceName)
    {
        return new LED();
    }
    else if (name == g_TIRFDeviceName)
    {
        return new TIRF();
    }

    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

static const std::vector<std::string> baudRates = { "115200", "28800", "19200", "9600"};

// Detect devices in the Hardware Configuration Wizard when you click "Scan Ports".
MM::DeviceDetectionStatus ASIDetectDevice(MM::Device& device, MM::Core& core, const std::string& port, double answerTimeoutMs)
{
    MM::DeviceDetectionStatus result = MM::Misconfigured;
    char savedTimeout[MM::MaxStrLength];

    try
    {
        // lowercase copy of port name
        std::string portLower = port;
        for (char& c : portLower)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        // skip invalid ports
        if (portLower.empty() || portLower == "undefined" || portLower == "unknown")
        {
            return result;
        }

        result = MM::CanNotCommunicate;

        // store the original timeout
        core.GetDeviceProperty(port.c_str(), "AnswerTimeout", savedTimeout);

        // device specific default communication parameters for ASIStage
        core.SetDeviceProperty(port.c_str(), MM::g_Keyword_Handshaking, "Off");
        core.SetDeviceProperty(port.c_str(), MM::g_Keyword_StopBits, "1");
        core.SetDeviceProperty(port.c_str(), "AnswerTimeout", std::to_string(answerTimeoutMs).c_str());
        core.SetDeviceProperty(port.c_str(), "DelayBetweenCharsMs", "0");

        MM::Device* pDevice = core.GetDevice(&device, port.c_str());

        // check all possible baud rates for the device
        for (const std::string& baudRate : baudRates)
        {
            core.SetDeviceProperty(port.c_str(), MM::g_Keyword_BaudRate, baudRate.c_str());
            pDevice->Initialize();
            core.PurgeSerial(&device, port.c_str());

            // check status using "/" command
            int ret = core.SetSerialCommand(&device, port.c_str(), "/", "\r");
            if (ret == DEVICE_OK)
            {
                char answer[MM::MaxStrLength];
                ret = core.GetSerialAnswer(&device, port.c_str(), MM::MaxStrLength, answer, "\r\n");
                if (ret != DEVICE_OK)
                {
                    LogDeviceError(device, core, ret);
                }
                else
                {
                    // success: device responded to the status command
                    result = MM::CanCommunicate;
                }
            }
            else
            {
                LogDeviceError(device, core, ret);
            }

            pDevice->Shutdown();

            if (result == MM::CanCommunicate)
            {
                break; // exit loop => we found the device
            }

            CDeviceUtils::SleepMs(10); // let GUI update
        }

        // restore timeout
        core.SetDeviceProperty(port.c_str(), "AnswerTimeout", savedTimeout);
    }
    catch (...)
    {
        core.LogMessage(&device, "Exception in ASIDetectDevice!", false);
    }

    return result;
}

// Helper function to log errors
void LogDeviceError(MM::Device& device, MM::Core& core, int errorCode) {
    char text[MM::MaxStrLength];
    device.GetErrorText(errorCode, text);
    core.LogMessage(&device, text, true);
}
