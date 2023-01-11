///////////////////////////////////////////////////////////////////////////////
// FILE:          MeadowlarkLC.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   MeadowlarkLC Device Adapter for Four Channel Digital Interface (D3050) and API family
//
// Copyright © 2009 - 2014, Marine Biological Laboratory
// 
// LICENSE (Berkeley Software Distribution License): Redistribution and use in source and binary forms,
// with or without modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
//    in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the Marine Biological Laboratory nor the names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as 
// representing official policies, either expressed or implied, of any organization.
//
// Developed at the Laboratory of Rudolf Oldenbourg at the Marine Biological Laboratory in Woods Hole, MA.
//
//
// AUTHOR: Amitabh Verma
//
// Notes: Refer MeadowlarkLC.cpp for ChangeLog, ToDo and other information

#ifndef _MEADOWLARKLC_H_
#define _MEADOWLARKLC_H_

#include "MMDevice.h"
#include "DeviceBase.h"

#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
#define ERR_PORT_CHANGE_FORBIDDEN 109
#define ERR_INVALID_DEVICE 102
#define ERR_INVALID_SERIAL_NUMBER 103
#define ERR_INVALID_LCSERIAL_NUMBER 104
#define ERR_INVALID_ACTIVE_LCS 105
#define ERR_INVALID_ACTIVATION_KEY 106
#define ERR_INVALID_LC_UNPAIRED 107
#define ERR_INVALID_LC_SELECTION 108
#define ERR_INVALID_LC_FILE 109
#define  flagsandattrs  0x40000000


class MeadowlarkLC : public CGenericBase<MeadowlarkLC>
{
public:
	MeadowlarkLC();
	~MeadowlarkLC();

	// Device API
	// ---------
	int Initialize();
	int Shutdown();	  

	void GetName(char* pszName) const;
	bool Busy();
	int GetVLCSerialAnswer (const char* portName, const char* term, std::string& ans);


	//      int Initialize(MM::Device& device, MM::Core& core);
	int DeInitialize() {initialized_ = false; return DEVICE_OK;};
	bool Initialized() {return initialized_;};

	// device discovery

	// action interface
	// ---------------
	int OnPort    (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBaud	(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBriefMode (MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnSerialNumber (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDevAdapterSerialNumber (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnActivationKey (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControllerType (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControllerLCType (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperature (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumDevices (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumTotalLCs (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumActiveLCs (MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetDesc (MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnDelay (MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnVoltage (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnRetardance (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnAbsRetardance (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnWavelength (MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnTneDuaration (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTneAmplitude (MM::PropertyBase* pProp, MM::ActionType eAct);

	//      int OnEpilogueL (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumPalEls (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPalEl (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnSendToMeadowlarkLC (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGetFromMeadowlarkLC (MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	HANDLE dev_Handle, pipe0, pipe1;
	UINT devcnt, counter;
	UINT USB_PID;
	GUID  theGUID;
	std::string cur_dev_name;      
	int cur_dev;
	double g_ControllerDeviceTypeVFac;
	double VoltageToRetFactor;
	double RetToVoltageFactor;
	double valTNE;
	double wavelength_; // the cached value	
	int numberofCurves; // default is 1, will change based on import
	int ArrayLength2;

	double ArrayLcVoltagesRetLoaded[1000][21]; // allow 7 curves import from csv
	double ArrayLcVoltagesRetCurve[1000][3];
	double ConfigData[250][3];
	double Wavelengths[11];

	std::vector<std::string> devNameList;
	std::string serialnum_;
	std::string activationKey_;
	std::string SystemSpecificActivationKey;
	std::string controllerType_;
	std::string controllerLCType_;
	std::string controllerLCType_Curve;
	std::string descriptionUnparsed_;
	std::string totalLCsMsg;
	std::string lcCurve_;

	double temperature_; // the cached value
	// Command exchange with MMCore
	std::string description_;

	bool initialized_;
	double answerTimeoutMs_;
	bool briefModeQ_;

	long numTotalDevices_;  // total number of LCs
	long numTotalLCs_;  // total number of LCs	  
	long numActiveLCs_;  // number of actively controlled LCs (the actively controlled LCs appear first in the list of retardance values in the L-command)
	double retardance_[26]; // retardance values of total number of LCs; I made the index 8, a high number unlikely to be exceeded by the variLC hardware
	double voltage_[26];
	std::string epilogueL_; // added at the end of every L command to account for uncontrolled LCs
	long numPalEls_;  // total number of palette elements
	std::string palEl_[99]; // array of palette elements, index is total number of elements
	double palette_[26][10]; // array of 26 palettes supporting 10 LCs
	//      std::string currRet_;

	double tneDuration_;
	double tneAmplitude_;
	bool interruptExercise_;

	std::string sendToMeadowlarkLC_;
	std::string getFromMeadowlarkLC_;
	MM::MMTime delay;
	MM::MMTime changedTime_; // is only required when writing to device and setting voltage not for other operations

	std::vector<double> getNumbersFromMessage(std::string variLCmessage, bool prefixQ);
	std::string RemoveChars(const std::string& source, const std::string& chars);
	std::string IntToString(int N);
	std::string DoubleToString(double N);
	int StringToInt(std::string str);	  
	double StringToDouble(std::string str);	  

	void hexconvert( char *text, unsigned char bytes[] );

	double VoltageToRetardance(double volt, long index);
	double RetardanceToVoltage(double retardance, long index);
	double RetardanceToVoltage(long wIndex, double AbsRetardance);

	void SendVoltageToDevice(int volt16bit, long index);
	void SetTneToDevice(int amplitude, double duration);

	double round(double number);
	double roundN(double number, int precision);
	double getValueFromArray(double val, int y, int curve_idx);
	void doExercise(int n);
	void import(std::string calibCurveFilename);
	void importConfig();
	void convertStringtoStringArray(std::string str);
	void exportCurve();
	void exportConfig();
	void clearConfig();
	void exportloadedCurve();
	void loadDefault();
	void generateCurve();
	void controllerLcTypeChange();

	void MeadowlarkLC::loadResource(int ID);

	void setSystemSpecificActivationKey(std::string SerialN);
	bool checkCalibFile(std::string strings);
	bool checkConfigFile(std::string strings);
};


#endif //_MEADOWLARKLC_H_
