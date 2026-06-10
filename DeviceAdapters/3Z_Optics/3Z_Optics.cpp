///////////////////////////////////////////////////////////////////////////////
// FILE:          3Z_Optics.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   3Z Optics Light Source driver
// COPYRIGHT:     3Z Optics
// LICENSE:       BSD license

#include "3Z_Optics.h"
#include "ModuleInterface.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

#define MODE_ADDR 0x20
#define CH1_INTENSITY_ADDR 0x31
#define CH1_SWITCH_ADDR 0x31
#define GLOBAL_INTENSITY_ADDR 0x30
#define GLOBAL_SWITCH_ADDR 0x30

enum
{
    MODE_GLOBAL = 1,
    MODE_INDEPENDENT,
    MODE_TTL
};

///////////////////////////////////////////////////////////////////////////////
// Static member initialization
///////////////////////////////////////////////////////////////////////////////
MMThreadLock Controller::lock_;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_DeviceName, MM::ShutterDevice, "3Z Optics Light Source");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    if (strcmp(deviceName, g_DeviceName) == 0)
    {
        return new Controller();
    }

    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// PollingThread implementation
///////////////////////////////////////////////////////////////////////////////
PollingThread::PollingThread(Controller& aController) :
    aController_(aController),
    stop_(false)
{
}

PollingThread::~PollingThread()
{
}

int PollingThread::svc(void)
{
    while (!stop_)
    {
        CDeviceUtils::SleepMs((long)aController_.pollIntervalMs_);

        if (!stop_)
        {
            MMThreadGuard guard(aController_.GetLock());
            aController_.PollDeviceStatus();
        }
    }
    return 0;
}

void PollingThread::Start()
{
    // when thread is restarted, make sure to reset the stop flag
    stop_ = false;
    activate();
}

///////////////////////////////////////////////////////////////////////////////
// Controller implementation
///////////////////////////////////////////////////////////////////////////////

Controller::Controller() :
    initialized_(false),
    shutterState_(false),
    deviceModelId_(0),
    globalIntensity_(0),
    globalSwitch_(false),
    currentMode_(1),
    pollIntervalMs_(2000.0),  // 2 second poll interval
    initializationComplete_(false),
    initializationInProgress_(false),
    globalSwitchUpdated_(false),
    globalIntensityUpdated_(false),
    modeUpdated_(false),
    mThread_(nullptr),
    channelSwitchUpdated_(),
    channelIntensityUpdated_()
{
    InitializeDefaultErrorMessages();

    // Set error messages
    SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Port cannot be changed after initialization");
    SetErrorText(ERR_DEVICE_NOT_FOUND, "Device model not found in configuration");
    SetErrorText(ERR_MODBUS_COMM_ERROR, "Modbus communication error");
    SetErrorText(ERR_INIT, "Initialization error");

    // Create pre-initialization properties (Name and Description will be created/updated in Initialize)

    // Port property
    CPropertyAction* pAct = new CPropertyAction(this, &Controller::OnPort);
    CreateProperty(MM::g_Keyword_Port, "", MM::String, false, pAct, true);
}

Controller::~Controller()
{
    Shutdown();
}

void Controller::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, g_DeviceName);
}

bool Controller::Busy()
{
    return false;
}

uint16_t Controller::CalculateCRC(const uint8_t* data, size_t length)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++)
    {
        crc ^= (uint16_t)data[i];
        for (size_t j = 0; j < 8; j++)
        {
            if (crc & 0x0001)
            {
                crc >>= 1;
                crc ^= 0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }
    return crc;
}

int Controller::SendModbusCommand(const std::vector<uint8_t>& request, std::vector<uint8_t>& response, int expectedResponseLength)
{
    if (port_.empty())
        return ERR_MODBUS_COMM_ERROR;

    MMThreadGuard guard(commLock_);

    // Clear port first
    int ret = PurgeComPort(port_.c_str());
    if (ret != DEVICE_OK)
        return ret;

    // Send request
    ret = WriteToComPort(port_.c_str(), &request[0], (unsigned long)request.size());
    if (ret != DEVICE_OK)
        return ret;

    // Wait a little longer for stable communication
    CDeviceUtils::SleepMs(100);

    // Read response - try multiple times if needed
    int maxAttempts = 3;
    for (int attempt = 0; attempt < maxAttempts; attempt++)
    {
        response.resize(expectedResponseLength);
        unsigned long read = 0;
        ret = ReadFromComPort(port_.c_str(), &response[0], (unsigned long)expectedResponseLength, read);

        if (ret == DEVICE_OK && read == (unsigned long)expectedResponseLength)
        {
            // Check CRC
            if (expectedResponseLength >= 2)
            {
                uint16_t crcCalc = CalculateCRC(&response[0], expectedResponseLength - 2);
                uint16_t crcRecv = (uint16_t)response[expectedResponseLength - 1] << 8 | (uint16_t)response[expectedResponseLength - 2];
                if (crcCalc == crcRecv)
                {
                    return DEVICE_OK;
                }
            }
            else
            {
                return DEVICE_OK;
            }
        }

        // If not last attempt, wait and retry
        if (attempt < maxAttempts - 1)
        {
            CDeviceUtils::SleepMs(50);
            PurgeComPort(port_.c_str());
        }
    }

    return ERR_MODBUS_COMM_ERROR;
}

int Controller::ReadInputRegister(int addr, uint16_t& value)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x04);  // Function code: Read Input Registers
    request.push_back((addr >> 8) & 0xFF); // Address high
    request.push_back(addr & 0xFF);        // Address low
    request.push_back(0x00);  // Number of registers high
    request.push_back(0x01);  // Number of registers low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, 7);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    value = ((uint16_t)response[3] << 8) | (uint16_t)response[4];
    return DEVICE_OK;
}

int Controller::WriteHoldingRegister(int addr, uint16_t value)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x06);  // Function code: Write Single Register
    request.push_back((addr >> 8) & 0xFF); // Address high
    request.push_back(addr & 0xFF);        // Address low
    request.push_back((value >> 8) & 0xFF); // Value high
    request.push_back(value & 0xFF);        // Value low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, 8);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    return DEVICE_OK;
}

int Controller::ReadHoldingRegister(int addr, uint16_t& value)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x03);  // Function code: Read Holding Registers
    request.push_back((addr >> 8) & 0xFF); // Address high
    request.push_back(addr & 0xFF);        // Address low
    request.push_back(0x00);  // Number of registers high
    request.push_back(0x01);  // Number of registers low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, 7);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    value = ((uint16_t)response[3] << 8) | (uint16_t)response[4];
    return DEVICE_OK;
}

int Controller::ReadMultipleHoldingRegisters(int startAddr, int count, std::vector<uint16_t>& values)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x03);  // Function code: Read Holding Registers
    request.push_back((startAddr >> 8) & 0xFF); // Address high
    request.push_back(startAddr & 0xFF);        // Address low
    request.push_back((count >> 8) & 0xFF);     // Number of registers high
    request.push_back(count & 0xFF);            // Number of registers low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    // Calculate expected response length
    int expectedLength = 5 + count * 2;  // addr(1) + func(1) + byteCount(1) + data(count*2) + crc(2)

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, expectedLength);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    // Parse response
    values.resize(count);
    for (int i = 0; i < count; i++)
    {
        int dataIndex = 3 + i * 2;
        values[i] = ((uint16_t)response[dataIndex] << 8) | (uint16_t)response[dataIndex + 1];
    }

    return DEVICE_OK;
}

int Controller::ReadSingleCoil(int addr, bool& on)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x01);  // Function code: Read Coils
    request.push_back((addr >> 8) & 0xFF); // Address high
    request.push_back(addr & 0xFF);        // Address low
    request.push_back(0x00);  // Number of coils high
    request.push_back(0x01);  // Number of coils low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, 6);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    // Response: address, function, byte count, coil status (1 byte)
    // Coil ON is 0xFF, OFF is 0x00
    on = (response[3] != 0);
    return DEVICE_OK;
}

int Controller::ReadMultipleCoils(int startAddr, int count, std::vector<bool>& values)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x01);  // Function code: Read Coils
    request.push_back((startAddr >> 8) & 0xFF); // Address high
    request.push_back(startAddr & 0xFF);        // Address low
    request.push_back((count >> 8) & 0xFF);     // Number of coils high
    request.push_back(count & 0xFF);            // Number of coils low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    // Calculate expected response length
    int byteCount = (count + 7) / 8;
    int expectedLength = 5 + byteCount;  // addr(1) + func(1) + byteCount(1) + data(byteCount) + crc(2)

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, expectedLength);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    // Parse response
    values.resize(count);
    for (int i = 0; i < count; i++)
    {
        int byteIndex = 3 + (i / 8);
        int bitIndex = i % 8;
        values[i] = ((response[byteIndex] >> bitIndex) & 0x01) != 0;
    }

    return DEVICE_OK;
}

int Controller::WriteSingleCoil(int addr, bool on)
{
    std::vector<uint8_t> request;
    request.push_back(0x01);  // Slave address
    request.push_back(0x05);  // Function code: Write Single Coil
    request.push_back((addr >> 8) & 0xFF); // Address high
    request.push_back(addr & 0xFF);        // Address low
    request.push_back(on ? 0xFF : 0x00);   // Value high (FF00=ON, 0000=OFF)
    request.push_back(0x00);               // Value low

    uint16_t crc = CalculateCRC(&request[0], request.size());
    request.push_back(crc & 0xFF);
    request.push_back((crc >> 8) & 0xFF);

    std::vector<uint8_t> response;
    int ret = SendModbusCommand(request, response, 8);
    if (ret != DEVICE_OK) return ret;

    if (response[1] & 0x80) // Check for exception
        return ERR_MODBUS_COMM_ERROR;

    return DEVICE_OK;
}

int Controller::ReadDeviceModel()
{
    uint16_t model;
    int ret = ReadInputRegister(0x01, model);
    if (ret != DEVICE_OK)
        return ret;

    deviceModelId_ = (int)model;
    return DEVICE_OK;
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(" \t\n\r");
    size_t last = str.find_last_not_of(" \t\n\r");
    if (first == string::npos)
        return "";
    return str.substr(first, (last - first + 1));
}

string readQuotedString(ifstream& file)
{
    string result;
    char c;
    // Skip leading whitespace
    while (file.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));
    if (c != '"')
        return "";

    while (file.get(c))
    {
        if (c == '"')
            break;
        if (c == '\\')
        {
            file.get(c);
            result += c;
        }
        else
        {
            result += c;
        }
    }
    return result;
}

int readInteger(ifstream& file)
{
    string result;
    char c;
    // Skip leading whitespace
    while (file.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));
    file.putback(c);

    while (file.get(c))
    {
        if (isdigit(static_cast<unsigned char>(c)))
        {
            result += c;
        }
        else
        {
            file.putback(c);
            break;
        }
    }

    if (result.empty())
        return 0;

    return stoi(result);
}

bool Controller::LoadDeviceConfig(int modelId)
{
    // Try multiple possible paths in order
    vector<string> possiblePaths;

    // 1. Try current working directory
    possiblePaths.push_back("models.json");
    possiblePaths.push_back("3z/models.json");

    // 2. Try user Documents folder using Windows API
    char userProfile[MAX_PATH] = { 0 };
    if (GetEnvironmentVariableA("USERPROFILE", userProfile, MAX_PATH))
    {
        possiblePaths.push_back(string(userProfile) + "\\Documents\\3z\\models.json");
        possiblePaths.push_back(string(userProfile) + "\\Documents\\models.json");
    }

    // 3. Try relative to Micro-Manager's directory
    possiblePaths.push_back("../../3z/models.json");
    possiblePaths.push_back("../../models.json");

    ifstream file;
    string foundPath;

    for (const string& path : possiblePaths)
    {
        file.open(path);
        if (file.is_open())
        {
            foundPath = path;
            break;
        }
    }

    if (foundPath.empty())
    {
        // Fall back to hardcoded default
        currentDevice_.name = "Unknown";
        currentDevice_.channels = { "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8" };
        currentDevice_.brightnessMin = 0;
        currentDevice_.brightnessMax = 100;
        return true;
    }

    string modelKey = to_string(modelId);
    bool foundModel = false;

    char c;
    // Skip until we find opening {
    while (file.get(c))
    {
        if (c == '{')
            break;
    }

    // Parse the JSON
    while (file.get(c))
    {
        if (c == '}')
            break;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            continue;

        if (c == '"')
        {
            file.putback(c);
            string key = readQuotedString(file);

            // Skip until :
            while (file.get(c) && c != ':');

            if (c == ':')
            {
                // Skip whitespace
                while (file.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));
                file.putback(c);

                if (c == '{') // Object
                {
                    file.get(c);

                    if (key == modelKey)
                    {
                        // Found our model, parse it
                        string name;
                        vector<string> channels;
                        int brightnessMin = 0;    // Default to 0
                        int brightnessMax = 100;  // Default to 100

                        // Parse the model object
                        while (file.get(c))
                        {
                            if (c == '}')
                                break;
                            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                                continue;

                            if (c == '"')
                            {
                                file.putback(c);
                                string propName = readQuotedString(file);

                                // Skip until :
                                while (file.get(c) && c != ':');

                                // Skip whitespace
                                while (file.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));
                                file.putback(c);

                                if (propName == "name")
                                {
                                    name = readQuotedString(file);
                                }
                                else if (propName == "BrightnessMin" || propName == "brightnessMin")
                                {
                                    brightnessMin = readInteger(file);
                                }
                                else if (propName == "BrightnessMax" || propName == "brightnessMax")
                                {
                                    brightnessMax = readInteger(file);
                                }
                                else if (propName == "channels")
                                {
                                    // Parse array
                                    if (file.get(c) && c == '[')
                                    {
                                        while (file.get(c))
                                        {
                                            if (c == ']')
                                                break;
                                            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',')
                                                continue;
                                            if (c == '"')
                                            {
                                                file.putback(c);
                                                string channel = readQuotedString(file);
                                                if (!channel.empty())
                                                    channels.push_back(channel);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Validate brightness range
                        if (brightnessMax < brightnessMin)
                        {
                            // Swap them if invalid
                            int temp = brightnessMin;
                            brightnessMin = brightnessMax;
                            brightnessMax = temp;
                        }

                        currentDevice_.name = name;
                        currentDevice_.channels = channels;
                        currentDevice_.brightnessMin = brightnessMin;
                        currentDevice_.brightnessMax = brightnessMax;
                        foundModel = true;
                        break;
                    }
                    else
                    {
                        // Skip this object
                        int depth = 1;
                        while (file.get(c) && depth > 0)
                        {
                            if (c == '{') depth++;
                            else if (c == '}') depth--;
                        }
                    }
                }
            }
        }
    }

    file.close();

    if (!foundModel)
    {
        currentDevice_.name = "Unknown";
        currentDevice_.channels = { "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8" };
        currentDevice_.brightnessMin = 0;
        currentDevice_.brightnessMax = 100;
    }

    return true;
}

int Controller::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    if (port_.empty())
        return ERR_INIT;

    // Mark that we're in initialization phase - prevents writing to hardware
    initializationInProgress_ = true;

    // Read device model from input register 0x01
    int ret = ReadDeviceModel();
    if (ret != DEVICE_OK)
    {
        initializationInProgress_ = false;
        return ret;
    }

    // Load device configuration
    if (!LoadDeviceConfig(deviceModelId_))
    {
        initializationInProgress_ = false;
        return ERR_DEVICE_NOT_FOUND;
    }

    // Create Name and Description properties with device name
    CreateStringProperty(MM::g_Keyword_Name, currentDevice_.name.c_str(), true);
    string desc = "3Z Optics " + currentDevice_.name;
    CreateStringProperty(MM::g_Keyword_Description, desc.c_str(), true);

    // Create device model and name properties
    CreateProperty(g_Prop_DeviceModel, CDeviceUtils::ConvertToString(deviceModelId_), MM::String, true);
    CreateProperty(g_Prop_DeviceName, currentDevice_.name.c_str(), MM::String, true);

    // Initialize channels
    channels_ = currentDevice_.channels;
    channelStates_.resize(channels_.size(), false);
    channelIntensities_.resize(channels_.size(), 0);
    channelSwitchUpdated_.resize(channels_.size(), false);
    channelIntensityUpdated_.resize(channels_.size(), false);

    // Create channel properties in the style of "385 Switch" and "385 Intensity"
    channelSwitchLookup_.clear();
    channelIntensityLookup_.clear();
    for (size_t i = 0; i < channels_.size(); i++)
    {
        // Channel Switch property (like "385 Switch")
        ostringstream switchName;
        switchName << channels_[i] << " Switch";
        CPropertyAction* pAct = new CPropertyAction(this, &Controller::OnChannelSwitch);
        CreateProperty(switchName.str().c_str(), "off", MM::String, false, pAct);
        AddAllowedValue(switchName.str().c_str(), "off");
        AddAllowedValue(switchName.str().c_str(), "on");
        channelSwitchLookup_[switchName.str()] = (int)i;

        // Channel Intensity property (like "385 Intensity"), range brightnessMin-brightnessMax
        ostringstream intensityName;
        intensityName << channels_[i] << " Intensity";
        pAct = new CPropertyAction(this, &Controller::OnChannelIntensity);
        CreateProperty(intensityName.str().c_str(), to_string(currentDevice_.brightnessMin).c_str(), MM::Integer, false, pAct);
        SetPropertyLimits(intensityName.str().c_str(), currentDevice_.brightnessMin, currentDevice_.brightnessMax);
        channelIntensityLookup_[intensityName.str()] = (int)i;
    }

    // Global Switch property
    CPropertyAction* pAct = new CPropertyAction(this, &Controller::OnGlobalSwitch);
    CreateProperty("Global Switch", "off", MM::String, false, pAct);
    AddAllowedValue("Global Switch", "off");
    AddAllowedValue("Global Switch", "on");

    // Global Intensity property, range brightnessMin-brightnessMax
    pAct = new CPropertyAction(this, &Controller::OnGlobalIntensity);
    CreateProperty("Global Intensity", to_string(currentDevice_.brightnessMin).c_str(), MM::Integer, false, pAct);
    SetPropertyLimits("Global Intensity", currentDevice_.brightnessMin, currentDevice_.brightnessMax);

    // Mode property
    pAct = new CPropertyAction(this, &Controller::OnMode);
    CreateProperty("Mode", "Global", MM::String, false, pAct);
    AddAllowedValue("Mode", "Global");
    AddAllowedValue("Mode", "Independent");
    AddAllowedValue("Mode", "TTL");

    // Refresh property - manual refresh button
    pAct = new CPropertyAction(this, &Controller::OnRefresh);
    CreateProperty("Refresh", "0", MM::Integer, false, pAct);
    SetPropertyLimits("Refresh", 0, 1);

    // Mark initialization complete before reading device state
    initializationComplete_ = true;

    // Read current device state from hardware instead of turning all off
    ret = ReadCurrentDeviceState();
    if (ret != DEVICE_OK)
    {
        initializationInProgress_ = false;
        return ret;
    }

    // Update all properties with initial values
    UpdatePropertiesFromDevice();

    initialized_ = true;

    // End initialization phase - now allows writing to hardware
    initializationInProgress_ = false;

    // Start polling thread after initialization is complete
    mThread_ = new PollingThread(*this);
    mThread_->Start();

    return DEVICE_OK;
}

int Controller::Shutdown()
{
    if (initialized_)
    {
        // Stop polling thread
        if (mThread_)
        {
            mThread_->Stop();
            mThread_->wait();
            delete mThread_;
            mThread_ = nullptr;
        }

        TurnAllOff();
        initialized_ = false;
    }
    return DEVICE_OK;
}

int Controller::SetOpen(bool open)
{
    MMThreadGuard guard(GetLock());

    // Skip writing to hardware during initialization
    if (initializationInProgress_)
    {
        shutterState_ = open;
        if (currentMode_ == MODE_GLOBAL)
        {
            globalSwitch_ = open;
        }

        return DEVICE_OK;
    }

    if (open)
    {
        shutterState_ = true;

        // Turn on global switch
        if (currentMode_ == MODE_GLOBAL)
        {
            globalSwitch_ = true;
            int ret = WriteSingleCoil(GLOBAL_SWITCH_ADDR, true);
            if (ret != DEVICE_OK)
            {
                shutterState_ = false;
                globalSwitch_ = false;
                return ret;
            }
        }

        // Apply channel states
        return ApplyChannelStates();
    }
    else
    {
        return TurnAllOff();
    }
}

int Controller::GetOpen(bool& open)
{
    MMThreadGuard guard(GetLock());

    open = shutterState_;
    return DEVICE_OK;
}

int Controller::ApplyChannelStates()
{
    MMThreadGuard guard(GetLock());

    // Skip writing to hardware during initialization
    if (initializationInProgress_)
    {
        return DEVICE_OK;
    }

    for (size_t i = 0; i < channelStates_.size(); i++)
    {
        // Set channel switch state with coil
        int ret = WriteSingleCoil(CH1_SWITCH_ADDR + (int)i, channelStates_[i] && shutterState_);
        if (ret != DEVICE_OK)
            return ret;
    }
    return DEVICE_OK;
}

int Controller::TurnAllOff()
{
    MMThreadGuard guard(GetLock());

    // Skip writing to hardware during initialization
    if (initializationInProgress_)
    {
        for (size_t i = 0; i < channels_.size(); i++)
        {
            channelStates_[i] = false;
        }
        shutterState_ = false;
        globalSwitch_ = false;
        return DEVICE_OK;
    }

    for (size_t i = 0; i < channels_.size(); i++)
    {
        // Turn off channel with coil
        int ret = WriteSingleCoil(CH1_SWITCH_ADDR + (int)i, false);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Turn off global switch with coil
    if (currentMode_ == MODE_GLOBAL)
    {
        int ret = WriteSingleCoil(GLOBAL_SWITCH_ADDR, false);
        if (ret != DEVICE_OK)
            return ret;
    }

    shutterState_ = false;
    globalSwitch_ = false;
    return DEVICE_OK;
}

int Controller::ReadDeviceStateByMode(int mode)
{
    MMThreadGuard guard(GetLock());

    int ret = DEVICE_OK;

    if (mode == 1) // Global mode
    {
        // Read global switch (coil 0x30) and global intensity (register 0x30)
        bool globalSwitch = false;
        ret = ReadSingleCoil(GLOBAL_SWITCH_ADDR, globalSwitch);
        if (ret != DEVICE_OK) return ret;
        globalSwitch_ = globalSwitch;
        shutterState_ = globalSwitch;

        uint16_t globalIntensity = 0;
        ret = ReadHoldingRegister(GLOBAL_INTENSITY_ADDR, globalIntensity);
        if (ret != DEVICE_OK) return ret;
        globalIntensity_ = (int)globalIntensity;
    }
    else if (mode == 2 || mode == 3) // Independent mode or TTL mode
    {
        // Read all channel coils and registers in batch
        int channelCount = (int)channels_.size();
        if (channelCount > 0)
        {
            // Read all channel coils at once
            std::vector<bool> coilValues;
            ret = ReadMultipleCoils(CH1_SWITCH_ADDR, channelCount, coilValues);
            if (ret != DEVICE_OK) return ret;
            for (int i = 0; i < channelCount; i++)
            {
                channelStates_[i] = coilValues[i];
            }

            // Read all channel registers at once
            std::vector<uint16_t> regValues;
            ret = ReadMultipleHoldingRegisters(CH1_INTENSITY_ADDR, channelCount, regValues);
            if (ret != DEVICE_OK) return ret;
            for (int i = 0; i < channelCount; i++)
            {
                channelIntensities_[i] = (int)regValues[i];
            }
        }
    }

    return DEVICE_OK;
}

int Controller::ReadCurrentDeviceState()
{
    MMThreadGuard guard(GetLock());

    // Read Mode (register 0x20)
    uint16_t modeVal = 0;
    int ret = ReadHoldingRegister(0x20, modeVal);
    if (ret != DEVICE_OK) return ret;
    currentMode_ = (int)modeVal;
    string modeStr;
    if (currentMode_ == MODE_GLOBAL)
        modeStr = "Global";
    else if (currentMode_ == MODE_INDEPENDENT)
        modeStr = "Independent";
    else if (currentMode_ == MODE_TTL)
        modeStr = "TTL";
    //else
    //    return DEVICE_ERR;

    // Read based on current mode
    ret = ReadDeviceStateByMode(currentMode_);
    if (ret != DEVICE_OK) return ret;

    // Update all properties with current values
    // Update Mode property
    SetProperty("Mode", modeStr.c_str());

    // Update global properties
    SetProperty("Global Switch", globalSwitch_ ? "on" : "off");
    SetProperty("Global Intensity", to_string(globalIntensity_).c_str());

    // Update all channel properties
    for (size_t i = 0; i < channels_.size(); i++)
    {
        ostringstream switchName;
        switchName << channels_[i] << " Switch";
        SetProperty(switchName.str().c_str(), channelStates_[i] ? "on" : "off");

        ostringstream intensityName;
        intensityName << channels_[i] << " Intensity";
        SetProperty(intensityName.str().c_str(), to_string(channelIntensities_[i]).c_str());
    }

    return DEVICE_OK;
}

int Controller::SetIntensity(int channel, int intensity)
{
    MMThreadGuard guard(GetLock());

    if (channel < 0 || channel >= (int)channelIntensities_.size())
        return DEVICE_ERR;

    channelIntensities_[channel] = intensity;

    if (shutterState_ && channelStates_[channel])
    {
        return WriteHoldingRegister(CH1_INTENSITY_ADDR + channel, (uint16_t)intensity);
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int Controller::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (initialized_)
        {
            pProp->Set(port_.c_str());
            return ERR_PORT_CHANGE_FORBIDDEN;
        }
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

int Controller::OnMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    if (eAct == MM::BeforeGet)
    {
        string modeStr;
        if (currentMode_ == MODE_GLOBAL)
            modeStr = "Global";
        else if (currentMode_ == MODE_INDEPENDENT)
            modeStr = "Independent";
        else if (currentMode_ == MODE_TTL)
            modeStr = "TTL";
        pProp->Set(modeStr.c_str());
        modeUpdated_ = false;
    }
    else if (eAct == MM::AfterSet)
    {
        // Skip writing to hardware during initialization
        if (initializationInProgress_)
        {
            return DEVICE_OK;
        }

        string newMode;
        pProp->Get(newMode);

        int newModeValue = 1;
        if (newMode == "Global")
            newModeValue = 1;
        else if (newMode == "Independent")
            newModeValue = 2;
        else if (newMode == "TTL")
            newModeValue = 3;

        int originalMode = currentMode_;
        currentMode_ = newModeValue;

        int ret = WriteHoldingRegister(0x20, (uint16_t)newModeValue);
        if (ret != DEVICE_OK)
        {
            currentMode_ = originalMode;
            string originalModeStr;
            if (originalMode == 1)
                originalModeStr = "Global";
            else if (originalMode == 2)
                originalModeStr = "Independent";
            else if (originalMode == 3)
                originalModeStr = "TTL";
            pProp->Set(originalModeStr.c_str());
            return ret;
        }

        // After setting new mode, read the current state for that mode
        ret = ReadDeviceStateByMode(newModeValue);
        if (ret != DEVICE_OK)
        {
            return ret;
        }

        // Update properties after reading new state
        UpdatePropertiesFromDevice();
    }
    return DEVICE_OK;
}

int Controller::OnChannelSwitch(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    string propName = pProp->GetName();
    int index = -1;
    auto it = channelSwitchLookup_.find(propName);
    if (it != channelSwitchLookup_.end())
    {
        index = it->second;
    }

    if (index < 0 || index >= (int)channelStates_.size())
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(channelStates_[index] ? "on" : "off");
        channelSwitchUpdated_[index] = false;
    }
    else if (eAct == MM::AfterSet)
    {
        // Skip writing to hardware during initialization
        if (initializationInProgress_)
        {
            return DEVICE_OK;
        }

        string newVal;
        pProp->Get(newVal);
        bool newState = (newVal == "on");

        // Save original state for rollback
        bool originalState = channelStates_[index];

        // Update the state
        channelStates_[index] = newState;

        // Send channel switch state to coil (address 0x31 + index)
        int ret = WriteSingleCoil(CH1_SWITCH_ADDR + index, newState && shutterState_);
        if (ret != DEVICE_OK)
        {
            // Send failed, roll back
            channelStates_[index] = originalState;
            pProp->Set(originalState ? "on" : "off");
            return ret;
        }
    }
    return DEVICE_OK;
}

int Controller::OnChannelIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    string propName = pProp->GetName();
    int index = -1;
    auto it = channelIntensityLookup_.find(propName);
    if (it != channelIntensityLookup_.end())
    {
        index = it->second;
    }

    if (index < 0 || index >= (int)channelIntensities_.size())
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)channelIntensities_[index]);
        channelIntensityUpdated_[index] = false;
    }
    else if (eAct == MM::AfterSet)
    {
        // Skip writing to hardware during initialization
        if (initializationInProgress_)
        {
            return DEVICE_OK;
        }

        long newVal;
        pProp->Get(newVal);
        int newIntensity = (int)newVal;

        // Validate and clamp the value
        if (newIntensity < currentDevice_.brightnessMin)
            newIntensity = currentDevice_.brightnessMin;
        else if (newIntensity > currentDevice_.brightnessMax)
            newIntensity = currentDevice_.brightnessMax;

        // Save original intensity for rollback
        int originalIntensity = channelIntensities_[index];

        // Update the intensity
        channelIntensities_[index] = newIntensity;

        // Always send to device regardless of channel switch state
        int ret = WriteHoldingRegister(CH1_INTENSITY_ADDR + index, (uint16_t)newIntensity);
        if (ret != DEVICE_OK)
        {
            // Send failed, roll back
            channelIntensities_[index] = originalIntensity;
            pProp->Set((long)originalIntensity);
            return ret;
        }
    }
    return DEVICE_OK;
}

int Controller::OnGlobalSwitch(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(globalSwitch_ ? "on" : "off");
        globalSwitchUpdated_ = false;
    }
    else if (eAct == MM::AfterSet)
    {
        // Skip writing to hardware during initialization
        if (initializationInProgress_)
        {
            return DEVICE_OK;
        }

        string pos;
        pProp->Get(pos);
        bool newGlobalSwitch = (pos == "on");

        // Save original state for rollback
        bool originalGlobalSwitch = globalSwitch_;

        globalSwitch_ = newGlobalSwitch;

        // Write global switch to coil (address 0x30)
        if (currentMode_ == MODE_GLOBAL)
        {
            int ret = WriteSingleCoil(GLOBAL_SWITCH_ADDR, newGlobalSwitch);
            if (ret != DEVICE_OK)
            {
                // Roll back
                globalSwitch_ = originalGlobalSwitch;
                pProp->Set(originalGlobalSwitch ? "on" : "off");
                return ret;
            }
        }

        // Also update shutter state
        shutterState_ = newGlobalSwitch;
    }
    return DEVICE_OK;
}

int Controller::OnGlobalIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long)globalIntensity_);
        globalIntensityUpdated_ = false;
    }
    else if (eAct == MM::AfterSet)
    {
        // Skip writing to hardware during initialization
        if (initializationInProgress_)
        {
            return DEVICE_OK;
        }
        if (currentMode_ != MODE_GLOBAL)
        {
            // Global Intensity only applies in Global mode
            return DEVICE_OK;
        }

        long pos;
        pProp->Get(pos);
        int newGlobalIntensity = (int)pos;

        // Validate and clamp the value
        if (newGlobalIntensity < currentDevice_.brightnessMin)
            newGlobalIntensity = currentDevice_.brightnessMin;
        else if (newGlobalIntensity > currentDevice_.brightnessMax)
            newGlobalIntensity = currentDevice_.brightnessMax;

        // Save original global intensity for rollback
        int originalGlobalIntensity = globalIntensity_;

        // Update global intensity
        globalIntensity_ = newGlobalIntensity;

        // Write global intensity to register 0x30
        int ret = WriteHoldingRegister(GLOBAL_INTENSITY_ADDR, (uint16_t)newGlobalIntensity);
        if (ret != DEVICE_OK)
        {
            // Roll back everything
            globalIntensity_ = originalGlobalIntensity;
            pProp->Set((long)originalGlobalIntensity);
            return ret;
        }
    }
    return DEVICE_OK;
}

int Controller::PollDeviceStatus()
{
    // Don't poll during initialization
    if (!initializationComplete_)
    {
        return DEVICE_OK;
    }

    // Read dirty bit coil 0x21
    bool dirtyBit = false;
    int ret = ReadSingleCoil(0x21, dirtyBit);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    // Only read registers if dirty bit is set
    if (dirtyBit)
    {
        // Read current mode first
        uint16_t modeVal = 0;
        ret = ReadHoldingRegister(0x20, modeVal);
        if (ret != DEVICE_OK) return ret;
        currentMode_ = (int)modeVal;

        // Read device state based on current mode
        ret = ReadDeviceStateByMode(currentMode_);
        if (ret != DEVICE_OK) return ret;

        // Update all properties
        UpdatePropertiesFromDevice();
    }

    return DEVICE_OK;
}

int Controller::ReadAllChannelRegisters()
{
    MMThreadGuard guard(GetLock());

    int channelCount = (int)channels_.size();
    if (channelCount > 0)
    {
        // Read all channel coils at once
        std::vector<bool> coilValues;
        int ret = ReadMultipleCoils(CH1_SWITCH_ADDR, channelCount, coilValues);
        if (ret != DEVICE_OK) return ret;
        for (int i = 0; i < channelCount; i++)
        {
            channelStates_[i] = coilValues[i];
        }

        // Read all channel registers at once
        std::vector<uint16_t> regValues;
        ret = ReadMultipleHoldingRegisters(CH1_INTENSITY_ADDR, channelCount, regValues);
        if (ret != DEVICE_OK) return ret;
        for (int i = 0; i < channelCount; i++)
        {
            channelIntensities_[i] = (int)regValues[i];
        }
    }
    return DEVICE_OK;
}

void Controller::UpdatePropertiesFromDevice()
{
    // Update Mode property
    modeUpdated_ = true;
    UpdateProperty("Mode");

    // Update global properties
    globalSwitchUpdated_ = true;
    UpdateProperty("Global Switch");

    globalIntensityUpdated_ = true;
    UpdateProperty("Global Intensity");

    // Update all channel properties
    for (size_t i = 0; i < channels_.size(); i++)
    {
        channelSwitchUpdated_[i] = true;
        ostringstream switchName;
        switchName << channels_[i] << " Switch";
        UpdateProperty(switchName.str().c_str());

        channelIntensityUpdated_[i] = true;
        ostringstream intensityName;
        intensityName << channels_[i] << " Intensity";
        UpdateProperty(intensityName.str().c_str());
    }

    // Notify UI that properties have changed
    OnPropertiesChanged();
}

int Controller::OnRefresh(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    MMThreadGuard guard(GetLock());

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(0L);
    }
    else if (eAct == MM::AfterSet)
    {
        // Read all device state
        ReadCurrentDeviceState();
        // Reset the property back to 0
        pProp->Set(0L);
    }
    return DEVICE_OK;
}
