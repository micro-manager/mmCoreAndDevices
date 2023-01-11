#include <string>

#define DEVICE_MAXNUMBER 8

const char* g_HubDevice_Name = "XLightMMHub";

const char* g_EmissionWheel_Name = "Emission wheel";
const char* g_EmissionWheel_Desc = "Emission filter wheel";

const char* g_DichroicWheel_Name = "Dichroic wheel";
const char* g_DichroicWheel_Desc = "Dichroic filter wheel";

const char* g_ExcitationWheel_Name = "Excitation wheel";
const char* g_ExcitationWheel_Desc = "Excitation filter wheel";

const char* g_SpinningSlider_Name = "Spinning slider";
const char* g_SpinningSlider_Desc = "Spinning slider";

const char* g_CameraSlider_Name = "Camera slider";
const char* g_CameraSlider_Desc = "Dual camera slider";

const char* g_SpinningMotor_Name = "Spinning motor";
const char* g_SpinningMotor_Desc = "Spinning motor";

const char* g_On = "On";
const char* g_Off = "Off";

const char* g_EmissionIrisDeviceName = "Emission Iris";
const char* g_EmissionIrisDeviceDescription = "Emission Iris";

const char* g_IlluminationIrisDeviceName = "Illumination Iris";
const char* g_IlluminationIrisDeviceDescription = "Illumination  Iris";

const char* g_IrisAperture = "Aperture";



const char* g_DichroicWheel_Dev_Desc = "Dichroic wheel device";
const char* g_EmissionWheel_Dev_Desc = "Emission wheel device";
const char* g_ExcitationWheel_Dev_Desc = "Excitation wheel device";
const char* g_SpinningSlider_Dev_Desc =  "Spinning slider device";
const char* g_CameraSlider_Dev_Desc = "Dual camera slider device";
const char* g_SpinningMotor_Dev_Desc = "Spinning motor device";
const char* g_EmissionIrisDevice_Dev_Desc = "Emission iris device";
const char* g_IlluminationIrisDevice_Dev_Desc = "Illumination iris device";
const char* g_HubDevice_Dev_Desc= "XLight Hub for MM";


// 30 secondi



std::string PositionLabels[DEVICE_MAXNUMBER]={"Emission pos.", "Dichroic pos.", "Excitation pos.", "Spinning pos.", "Slider pos.", "Motor", "", ""};
std::string DevicesName[DEVICE_MAXNUMBER]={"Emission wheel", "Dichroic wheel", "Excitation wheel", "Spinning slider", "Camera slider", "Spinning motor", "Emission Iris", "Illumination Iris"};
std::string DevicesDesc[DEVICE_MAXNUMBER]={"Emission filter wheel", "Dichroic filter wheel", "Excitation filter wheel", "Spinning slider", "Dual camera slider", "Spinning motor", "Emission Iris", "Illumination Iris"};


std::string CMDPrefix[DEVICE_MAXNUMBER]={"B", "C", "A","D", "P", "N", "V", "J"};
bool DeviceOnline[DEVICE_MAXNUMBER]={true, true, true,true, true, true, true, true}; //device connected
long MaxPositions[DEVICE_MAXNUMBER]; // device range
long InitialPositions[DEVICE_MAXNUMBER]; // initial position of devices

const char* NoHubError = "Parent Hub not defined.";
