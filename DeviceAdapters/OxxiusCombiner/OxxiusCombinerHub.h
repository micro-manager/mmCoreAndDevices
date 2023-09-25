#pragma once
#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/ModuleInterface.h"
#include <cstdlib>
#include <string>
#include <map>
#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>  // For system time
#include <iomanip> // For formatting time
#include <sstream> // For string streams
using namespace std;

//For Obis
#ifdef WIN32
	#include <windows.h>
#endif
#include "../../MMDevice/DeviceUtils.h"
#include "FixSnprintf.h"
#include <algorithm>
#include <math.h>
//

//
class OxxiusLBX; // advance declaration
class OxxiusLCX; // advance declaration
class CoherentObis; //advance declaration
class Cobolt08_01; //advance declaration
class OxxiusShutter; // advance declaration
class OxxiusMDual; // advance declaration
class OxxiusFlipMirror; // advance declaration

//
#define	MAX_NUMBER_OF_SLOTS	6
#define	RCV_BUF_LENGTH 256
#define	NO_SLOT 0

//////////////////////////////////////////////////////////////////////////////
// Error codes for LBX and LCX
//
#define ERR_PORT_CHANGE_FORBIDDEN	101
#define ERR_NO_PORT_SET				102
#define ERR_COMBINER_NOT_FOUND	    201
#define ERR_UNSUPPORTED_VERSION	    202

//////////////////////////////////////////////////////////////////////////////
// Error codes for Obis
//
#define OBIS_ERR_PORT_CHANGE_FORBIDDEN    10004
#define OBIS_ERR_DEVICE_NOT_FOUND         10005

#define OBIS_POWERCONVERSION              1000 //convert the power into mW from the W it wants the commands in

//////////////////////////////////////////////////////////////////////////////
// Miscellaneous definitions
//
// Use the name 'return_value' that is unlikely to appear within 'result'.
#define RETURN_ON_MM_ERROR( result ) do { \
   int return_value = (result); \
   if (return_value != DEVICE_OK) { \
      return return_value; \
   } \
} while (0)




//////////////////////////////////////////////////////////////////////////////
// Defining device adaptaters
//


class OxxiusCombinerHub : public HubBase<OxxiusCombinerHub>
{
public:
	OxxiusCombinerHub();
	~OxxiusCombinerHub();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();
	int DetectInstalledDevices();
	unsigned int GetNumberOfInstalledDevices() { return installedDevices_; };
	//	MM::DeviceDetectionStatus DetectDevice(void);
	bool SupportsDeviceDetection(void) { return true; };


	// Property handlers
	int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnSerialNumber(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnInterlock(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnEmissionKey(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTemperature(MM::PropertyBase* pPropt, MM::ActionType eAct);

	// Custom interface for child devices
//	bool IsPortAvailable() {return portAvailable_;}
	int QueryCommand(MM::Device* device, MM::Core* core, const unsigned int destinationSlot, const char* command, bool adco);
	int ParseforBoolean(bool& destBoolean);
	int ParseforFloat(float& destFloat);
	int ParseforInteger(unsigned int& destInteger);
	int ParseforString(string& destString);
	int ParseforVersion(unsigned int& Vval);
	int ParseforPercent(double& Pval);
	int ParseforTemperature(float& Tval);
	int ParseforChar(char* Nval);

	int ParseforDouble(double& destDouble);
	// int TempAdminInt(const char* com);
	// int TempAdminString(int com, string &res);

	bool GetAOMpos1(unsigned int slot);
	bool GetAOMpos2(unsigned int slot);
	bool GetMPA(unsigned int slot);
	int GetObPos();

private:
	void LogError(int id, MM::Device* device, MM::Core* core, const char* functionName);

	string port_;
	bool initialized_;
	unsigned int installedDevices_;

	string type_;
	float maxtemperature_;

	string serialAnswer_;

	string serialNumber_;
	bool interlockClosed_;
	bool keyActivated_;

	unsigned int AOM1pos_;
	unsigned int AOM2pos_;
	unsigned int mpa[7];
	unsigned int obPos_;
};
