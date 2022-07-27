///////////////////////////////////////////////////////////////////////////////
// FILE:          MeadowlarkLC.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   MeadowlarkLC Device Adapter for Meadowlark Optics liquid crystal controllers
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

#include <string>
#include <map>

#include "DeviceBase.h"

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
	int GetVLCSerialAnswer(const char* portName, const char* term, std::string& ans);


	//      int Initialize(MM::Device& device, MM::Core& core);
	int DeInitialize() { initialized_ = false; return DEVICE_OK; };
	bool Initialized() { return initialized_; };
	void loadResource(int ID);

	// device discovery

	// action interface
	// ---------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBaud(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBriefMode(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDevAdapterVersionNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnActivationKey(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControllerType(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControllerLCType(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumDevices(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumTotalLCs(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumActiveLCs(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetDesc(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnRetardance(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnAbsRetardance(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnWavelength(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnTneDuaration(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTneAmplitude(MM::PropertyBase* pProp, MM::ActionType eAct);

	//      int OnEpilogueL (MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumPalEls(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPalEl(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnSendToMeadowlarkLC(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGetFromMeadowlarkLC(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	HANDLE dev_Handle, pipe0, pipe1;
	UINT devcnt, i;
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

	void hexconvert(char* text, unsigned char bytes[]);

	double VoltageToRetardance(double volt, long index);
	double RetardanceToVoltage(double retardance, long index);
	double RetardanceToVoltage(long wIndex, double AbsRetardance);
	double GetVoltage(long index);

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

	void setSystemSpecificActivationKey(std::string SerialN);
	bool checkCalibFile(std::string strings);
	bool checkConfigFile(std::string strings);
};


const int ArrayLengthDefault = 198;
// LC Calibration curve Voltage(mV) vs Absolute Retardance(nm)
const double ArrayDefaultLcVoltagesRet[ArrayLengthDefault][3] = {
	{0,1284.343,1284.343},
	{200,1284.411,1284.411},
	{400,1284.0869,1284.0869},
	{600,1283.8796,1283.8796},
	{800,1283.3625,1283.3625},
	{800.0002,1283.2228,1283.2228},
	{800.0005,1283.3013,1283.3013},
	{800.0013,1283.415,1283.415},
	{800.0025,1283.3468,1283.3468},
	{800.0045,1283.4783,1283.4783},
	{800.0074,1283.3376,1283.3376},
	{800.0117,1283.3345,1283.3345},
	{800.0179,1283.4308,1283.4308},
	{800.0264,1283.496,1283.496},
	{800.0377,1283.4891,1283.4891},
	{800.0527,1283.4348,1283.4348},
	{800.0721,1283.7001,1283.7001},
	{800.0969,1283.6556,1283.6556},
	{800.128,1283.7085,1283.7085},
	{800.1666,1283.7365,1283.7365},
	{800.214,1283.7697,1283.7697},
	{800.2716,1283.6506,1283.6506},
	{800.3411,1283.7167,1283.7167},
	{800.424,1283.6982,1283.6982},
	{800.5223,1283.7743,1283.7743},
	{800.638,1283.8627,1283.8627},
	{800.7734,1283.7518,1283.7518},
	{800.9308,1283.8488,1283.8488},
	{801.113,1283.8833,1283.8833},
	{801.3225,1283.8304,1283.8304},
	{801.5625,1283.8893,1283.8893},
	{801.8361,1283.8992,1283.8992},
	{802.1468,1283.9257,1283.9257},
	{802.4982,1283.9562,1283.9562},
	{802.8941,1284.0687,1284.0687},
	{803.3387,1284.1172,1284.1172},
	{803.8364,1284.0798,1284.0798},
	{804.3917,1284.0042,1284.0042},
	{805.0095,1284.047,1284.047},
	{805.6949,1284.0695,1284.0695},
	{806.4534,1284.1594,1284.1594},
	{807.2906,1284.1851,1284.1851},
	{808.2125,1284.1339,1284.1339},
	{809.2255,1284.2399,1284.2399},
	{810.336,1284.2579,1284.2579},
	{811.5509,1284.3578,1284.3578},
	{812.8775,1284.1702,1284.1702},
	{814.3233,1284.2469,1284.2469},
	{815.8961,1284.1438,1284.1438},
	{817.6041,1284.283,1284.283},
	{819.4559,1284.301,1284.301},
	{821.4603,1284.2782,1284.2782},
	{823.6266,1284.1348,1284.1348},
	{825.9644,1284.2441,1284.2441},
	{828.4838,1284.1818,1284.1818},
	{831.1949,1284.3066,1284.3066},
	{834.1088,1284.2123,1284.2123},
	{837.2363,1284.1847,1284.1847},
	{840.5892,1284.2885,1284.2885},
	{844.1794,1284.2258,1284.2258},
	{848.0192,1284.1924,1284.1924},
	{852.1216,1284.3458,1284.3458},
	{856.4996,1284.1655,1284.1655},
	{861.1669,1284.0703,1284.0703},
	{866.1377,1284.145,1284.145},
	{871.4266,1284.1002,1284.1002},
	{877.0485,1284.0598,1284.0598},
	{883.0189,1284.0197,1284.0197},
	{889.3539,1283.9844,1283.9844},
	{896.0698,1283.9204,1283.9204},
	{903.1836,1283.9547,1283.9547},
	{910.7126,1283.9744,1283.9744},
	{918.675,1283.9253,1283.9253},
	{927.0889,1283.9131,1283.9131},
	{935.9734,1283.7408,1283.7408},
	{945.3481,1283.6477,1283.6477},
	{955.2328,1283.6884,1283.6884},
	{965.6481,1283.6045,1283.6045},
	{976.6151,1283.4708,1283.4708},
	{988.1553,1283.4731,1283.4731},
	{1000.2911,1283.4991,1283.4991},
	{1013.0452,1283.2888,1283.2888},
	{1026.4408,1283.2068,1283.2068},
	{1040.502,1282.854,1282.854},
	{1055.2531,1282.8246,1282.8246},
	{1070.7194,1282.4509,1282.4509},
	{1086.9263,1282.1366,1282.1366},
	{1103.9004,1281.6805,1281.6805},
	{1121.6683,1281.3497,1281.3497},
	{1140.2579,1280.9851,1280.9851},
	{1159.6971,1280.6768,1280.6768},
	{1180.0148,1280.2689,1280.2689},
	{1201.2404,1279.3618,1279.3618},
	{1223.4039,1278.7089,1278.7089},
	{1246.5363,1277.9419,1277.9419},
	{1270.6688,1276.0315,1276.0315},
	{1295.8336,1274.8234,1274.8234},
	{1322.0635,1272.7963,1272.7963},
	{1349.3917,1270.1302,1270.1302},
	{1377.8527,1266.1652,1266.1652},
	{1407.4811,1260.8419,1260.8419},
	{1438.3125,1253.3003,1253.3003},
	{1470.3833,1244.5061,1244.5061},
	{1503.7303,1232.5726,1232.5726},
	{1538.3914,1217.3304,1217.3304},
	{1574.4049,1198.2084,1198.2084},
	{1611.8101,1176.7759,1176.7759},
	{1650.6469,1154.4043,1154.4043},
	{1690.9559,1128.3533,1128.3533},
	{1732.7789,1089,1089},
	{1821.1359,1047.5177,1047.5177},
	{1867.7567,1016.2104,1016.2104},
	{1916.0651,984.7769,984.7769},
	{1966.1063,953.2874,953.2874},
	{2017.9265,920.6101,920.6101},
	{2071.573,887.09,887.09},
	{2127.0933,851.7007,851.7007},
	{2184.5361,816.75,816.75},
	{2305.3887,754.0078,754.0078},
	{2368.8997,721.3992,721.3992},
	{2434.5366,689.3527,689.3527},
	{2502.3521,658.8939,658.8939},
	{2572.3999,628.5602,628.5602},
	{2644.7349,599.2983,599.2983},
	{2719.4126,569.1844,569.1844},
	{2796.4893,544.5,544.5},
	{2958.0703,491.2751,491.2751},
	{3042.6919,468.1365,468.1365},
	{3129.9478,446.146,446.146},
	{3219.8982,425.1326,425.1326},
	{3312.6057,405.4332,405.4332},
	{3408.1331,387.0562,387.0562},
	{3506.5439,369.5298,369.5298},
	{3607.9033,353.7877,353.7877},
	{3712.2771,338.264,338.264},
	{3819.7314,324.5506,324.5506},
	{3930.3347,308.7356,308.7356},
	{4044.1553,296.194,296.194},
	{4161.2627,284.466,284.466},
	{4281.728,272.25,272.25},
	{4533.0195,249.2416,249.2416},
	{4663.9917,240.8623,240.8623},
	{4798.6152,231.4884,231.4884},
	{4936.9644,221.9211,221.9211},
	{5079.1172,214.1089,214.1089},
	{5225.1514,206.1059,206.1059},
	{5375.145,199.4312,199.4312},
	{5529.1787,192.4627,192.4627},
	{5687.3335,186.2214,186.2214},
	{5849.6919,179.7628,179.7628},
	{6016.3364,173.8859,173.8859},
	{6187.3521,168.3429,168.3429},
	{6362.8232,162.9014,162.9014},
	{6542.8379,157.5549,157.5549},
	{6727.4824,152.6496,152.6496},
	{6916.8462,147.6369,147.6369},
	{7111.019,143.0052,143.0052},
	{7310.0918,138.6415,138.6415},
	{7514.1567,134.3855,134.3855},
	{7723.3071,130.3516,130.3516},
	{7937.6372,126.2824,126.2824},
	{8157.2432,122.406,122.406},
	{8382.2217,118.8456,118.8456},
	{8612.6699,115.4299,115.4299},
	{8848.6885,111.6104,111.6104},
	{9090.377,108.5394,108.5394},
	{9337.8359,105.2933,105.2933},
	{9591.1699,102.3992,102.3992},
	{9850.4814,99.3358,99.3358},
	{10115.876,96.4839,96.4839},
	{10387.4609,93.7825,93.7825},
	{10665.3428,91.202,91.202},
	{10949.6318,88.5837,88.5837},
	{11240.4365,86.0813,86.0813},
	{11537.8691,83.6826,83.6826},
	{11842.042,81.2935,81.2935},
	{12153.0684,79.0205,79.0205},
	{12471.0654,76.9506,76.9506},
	{12796.1484,75.0336,75.0336},
	{13128.4346,72.9454,72.9454},
	{13468.0439,70.8509,70.8509},
	{13815.0967,69.1639,69.1639},
	{14169.7139,67.2585,67.2585},
	{14532.0195,65.5513,65.5513},
	{14902.1367,63.9257,63.9257},
	{15280.1924,61.706,61.706},
	{15666.3135,60.4764,60.4764},
	{16060.627,58.9538,58.9538},
	{16463.2637,57.8524,57.8524},
	{16874.3535,55.6492,55.6492},
	{17294.0312,54.4416,54.4416},
	{17722.4277,53.0858,53.0858},
	{18159.6816,51.6723,51.6723},
	{18605.9258,50.635,50.635},
	{19061.3008,49.3032,49.3032},
	{19525.9453,47.6384,47.6384},
	{20000,46.83,46.83}

};


#endif //_MEADOWLARKLC_H_
