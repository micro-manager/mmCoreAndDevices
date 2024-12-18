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

MM::DeviceDetectionStatus ASICheckSerialPort(MM::Device& device, MM::Core& core, std::string portToCheck, double answerTimeoutMs)
{
    // all conditions must be satisfied...
    MM::DeviceDetectionStatus result = MM::Misconfigured;
    char answerTO[MM::MaxStrLength];
    try
    {
        std::string portLowerCase = portToCheck;
        for (std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
        {
            *its = (char)tolower(*its);
        }
        if (0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown"))
        {
            result = MM::CanNotCommunicate;
            core.GetDeviceProperty(portToCheck.c_str(), "AnswerTimeout", answerTO);
            // device specific default communication parameters for ASI Stage
            core.SetDeviceProperty(portToCheck.c_str(), MM::g_Keyword_Handshaking, "Off");
            core.SetDeviceProperty(portToCheck.c_str(), MM::g_Keyword_StopBits, "1");
            std::ostringstream too;
            too << answerTimeoutMs;
            core.SetDeviceProperty(portToCheck.c_str(), "AnswerTimeout", too.str().c_str());
            core.SetDeviceProperty(portToCheck.c_str(), "DelayBetweenCharsMs", "0");
            MM::Device* pS = core.GetDevice(&device, portToCheck.c_str());
            std::vector< std::string> possibleBauds;
            possibleBauds.push_back("9600");
            possibleBauds.push_back("115200");
            for (std::vector< std::string>::iterator bit = possibleBauds.begin(); bit != possibleBauds.end(); ++bit)
            {
                core.SetDeviceProperty(portToCheck.c_str(), MM::g_Keyword_BaudRate, (*bit).c_str());
                pS->Initialize();
                core.PurgeSerial(&device, portToCheck.c_str());
                // check status
                const char* command = "/";
                int ret = core.SetSerialCommand(&device, portToCheck.c_str(), command, "\r");
                if (DEVICE_OK == ret)
                {
                    char answer[MM::MaxStrLength];
                    ret = core.GetSerialAnswer(&device, portToCheck.c_str(), MM::MaxStrLength, answer, "\r\n");
                    if (DEVICE_OK != ret)
                    {
                        char text[MM::MaxStrLength];
                        device.GetErrorText(ret, text);
                        core.LogMessage(&device, text, true);
                    }
                    else
                    {
                        // to succeed must reach here....
                        result = MM::CanCommunicate;
                    }
                }
                else
                {
                    char text[MM::MaxStrLength];
                    device.GetErrorText(ret, text);
                    core.LogMessage(&device, text, true);
                }
                pS->Shutdown();
                if (MM::CanCommunicate == result)
                {
                    break;
                }
                else
                {
                    // try to yield to GUI
                    CDeviceUtils::SleepMs(10);
                }
            }
            // always restore the AnswerTimeout to the default
            core.SetDeviceProperty(portToCheck.c_str(), "AnswerTimeout", answerTO);
        }
    }
    catch (...)
    {
        core.LogMessage(&device, "Exception in DetectDevice!", false);
    }
    return result;
}
