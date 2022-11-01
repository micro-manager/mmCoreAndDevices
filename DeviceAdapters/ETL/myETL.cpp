#include "MMDevice.h"
#include "myETL.h"
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <sstream>


// myETL 
const char* myETLName = "ETL";
const char* carriage_return = "\r";
const char* end_of_line = "\r\n";
const char* g_PropertyMaxImA = "MaxI_mA"; // current max
const char* g_PropertyMinImA = "MinI_mA"; // current min

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(myETLName, MM::GenericDevice, "Optotune ETL");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
   {
      return 0;
   }
   if (strcmp(deviceName, myETLName) == 0)
   {
      // create myETL
      myETL* pMyETL = new myETL(myETLName);// myETL(myETLName);
      return pMyETL;
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// myETL implementation
///////////////////////////////////////////////////////////////////////////////

myETL::myETL(const char* name) :
   initialized_(false),
   port_("Undefined"),
   name_(name),
   error_(false)
{
   assert(strlen(name) < (unsigned int)MM::MaxStrLength);

   // create pre-initialization properties
   // ------------------------------------

   // Name
   CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);

   // Description
   CreateProperty(MM::g_Keyword_Description, "Optotune Electric Tunable Lens", MM::String, true);

   // Port
   CPropertyAction* pAct = new CPropertyAction(this, &myETL::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   //// Max & Min
   CreateProperty(g_PropertyMaxImA, "293", MM::Float, false, 0, true);
   CreateProperty(g_PropertyMinImA, "-293", MM::Float, false, 0, true);

   EnableDelay(); // signals that the delay setting will be used

}

myETL::~myETL()
{
   Shutdown();
}


bool myETL::Busy()
{
   return false;
}

int myETL::Initialize()
{
   this->LogMessage("myETL::Initialize()");

   GeneratePropertyCurrent();

   Send("Start");

   return 0;
}

void myETL::GetName(char* name) const
{
   assert(name_.length() < CDeviceUtils::GetMaxStringLength());
   CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

int myETL::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return 0;
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

void myETL::GeneratePropertyCurrent()
{
   CPropertyAction* pAct = new CPropertyAction(this, &myETL::OnCurrent);
   CreateProperty("Current-mA", "0", MM::Float, false, pAct);

   double upperLimit = 293.0;
   double lowerLimit = -293.0;
   SetPropertyLimits("Current-mA", lowerLimit, upperLimit);

}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////


int myETL::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         // revert
         pProp->Set(port_.c_str());
         return -1;
      }

      pProp->Get(port_);
   }

   return 0;
}

int myETL::OnCurrent(MM::PropertyBase* pProp, MM::ActionType eAct)
{

   double current;
   if (eAct == MM::BeforeGet)
   {
      GetCurrent(current);
      pProp->Set(current);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(current);
      SetCurrent(current);
   }

   return 0;
}


///////////////////////////////////////////////////////////////////////////////
// Utility methods
///////////////////////////////////////////////////////////////////////////////
uint16_t myETL::BitReflect(uint16_t data, unsigned char nbits) 
{
   uint16_t output = 0;
   for (unsigned char i = 0; i < nbits; ++i) 
   {
      if (data & 1)
         output |= 1 << ((nbits - 1) - i);
      data >>= 1;
   }
   return output;
}

uint16_t myETL::Crc16Ibm(char* data, unsigned int data_len) 
{
   uint16_t crc = 0;

   if (data_len == 0)
      return 0;

   for (unsigned int i = 0; i < data_len; ++i) 
   {
      uint16_t dbyte = BitReflect(data[i], 8);
      crc ^= dbyte << 8;

      for (unsigned char j = 0; j < 8; ++j) 
      {
         uint16_t mix = crc & 0x8000;
         crc = (crc << 1);
         if (mix)
            crc = crc ^ 0x8005;
      }
   }

   return BitReflect(crc, 16);
}

void myETL::StringToHexa(char* input, char* output)
{
   int loop;
   int i;

   i = 0;
   loop = 0;
   
   while (input[loop] != '\0')
   {
      sprintf_s((char*)(output + i), 3, "%02X", input[loop]);
      std::string mytext(reinterpret_cast<char*>(output));
      loop += 1;
      i += 2;
   }
   //insert NULL at the end of the output string
   output[i++] = '\0';

}

void myETL::DecToHexa(int n, char* output)
{
   // char array to store hexadecimal number
   char hexaDeciNum[100];

   // counter for hexadecimal number array
   uint32_t num = n; // format très important pour les valeurs négatives
   int i = 0;
   while (num != 0)
   {
      // temporary variable to store remainder
      int temp = 0;

      // storing remainder in temp variable.
      temp = num % 16;

      // check if temp < 10
      if (temp < 10)
      {
         hexaDeciNum[i] = temp + 48; // 48 en decimal = 0 en char ?
         i++;
      }
      else
      {
         hexaDeciNum[i] = temp + 55;
         i++;
      }

      num = num / 16;
   }

   int k = 0;

   // on s'attend à un hexa en 2 octets
   if (i < 4)
   {
      output[0] = '0';
      k++;
   }
   if (i < 3)
   {
      output[1] = '0';
      k++;
   }
   if (i < 2)
   {
      output[2] = '0';
      k++;
   }
   if (i < 1)
   {
      output[3] = '0';
      k++;
   }

   /*if (i % 2 != 0)
   {
      output[0] = '0';
      k++;
   }*/

   // printing hexadecimal number array in reverse order
   if (i > 4) // hexa en 2 octets
      i = 4;
   for (int j = i - 1; j >= 0; j--)
   {
      output[k] = hexaDeciNum[j];
      this->LogMessage("output dectohexa : " + std::to_string(i) + " "+ std::to_string(j) + " " +std::to_string(output[k]));
      k++;
   }
   output[k] = '\0';

}

unsigned char* myETL::CreateAndSendCurrentCommand(double current)
{
   char channel_data[5];
   char data[] = "Aw";
   StringToHexa(data, channel_data);

   char number_data[5];
   short coded_current = (short)((int)(current / 293.0 * 4096.0)); // 293 a la place de 292  ensuite
   DecToHexa(coded_current, number_data);
   this->LogMessage("current : " + std::to_string(coded_current) + " - size of hexa :" + std::to_string(sizeof(number_data)));

   char before_crc[9];
   memcpy_s(before_crc, _countof(before_crc), channel_data, _countof(channel_data));
   memcpy(before_crc + _countof(channel_data) - 1, number_data, sizeof(number_data));

   char before_crc_cut[12];
   before_crc_cut[0] = before_crc[0];
   before_crc_cut[1] = before_crc[1];
   before_crc_cut[2] = ' ';
   before_crc_cut[3] = before_crc[2];
   before_crc_cut[4] = before_crc[3];
   before_crc_cut[5] = ' ';
   before_crc_cut[6] = before_crc[4];
   before_crc_cut[7] = before_crc[5];
   before_crc_cut[8] = ' ';
   before_crc_cut[9] = before_crc[6];
   before_crc_cut[10] = before_crc[7];
   before_crc_cut[11] = ' ';

   std::vector<std::string> destination;
   std::string part;
   std::stringstream token;
   token << before_crc_cut;

   while (getline(token, part, ' ')) 
      destination.push_back(part);

   char* char_arr;
   std::string str_obj(destination.at(0));
   char_arr = &str_obj[0];
   char output_uc[4];
   char addressBytes[2];
   for (int i = 0; i < 4; i++) 
   {
      str_obj = destination[i];

      this->LogMessage("str_obj : "+str_obj);

      sscanf_s(char_arr, "%02hhx%02hhx", &addressBytes[0], &addressBytes[1]); // not null terminated
      output_uc[i] = addressBytes[0];
      this->LogMessage("output_uc : " + std::to_string(output_uc[i]));
   }
   uint16_t crc_ibm = Crc16Ibm((char*)output_uc, sizeof(output_uc));

   char crc_data[5];
   DecToHexa(crc_ibm, crc_data);
   char crc_data_invert[6];
   crc_data_invert[0] = crc_data[2];
   crc_data_invert[1] = crc_data[3];
   crc_data_invert[2] = ' ';
   crc_data_invert[3] = crc_data[0];
   crc_data_invert[4] = crc_data[1];
   crc_data_invert[5] = ' ';
   token = std::stringstream();
   std::vector<std::string> destination2;
   token << crc_data_invert;
   while (getline(token, part, ' ')) destination2.push_back(part);
   
   char crc_uc[2];
   char* char_arr2;
   std::string str_obj2(destination2.at(0));
   char_arr2 = &str_obj2[0];
   char addressBytes2[2];
   for (int i = 0; i < 2; i++)
   {
      str_obj2 = destination2[i];
      sscanf_s(char_arr2, "%02hhx%02hhx", &addressBytes2[0], &addressBytes2[1]); // not null terminated
      crc_uc[i] = addressBytes2[0];
      this->LogMessage("crc_uc :" + std::to_string(crc_uc[i]));
   }

   char after_crc_uc[6];
   memcpy_s(after_crc_uc, sizeof(after_crc_uc), output_uc, _countof(output_uc)); // output uc = sans le crc
   memcpy(after_crc_uc + _countof(output_uc), crc_uc, sizeof(crc_uc));
  
   unsigned char cmd_hex[6];
   
   for (int i = 0; i < 6; i++)
   {
      cmd_hex[i] = (unsigned)after_crc_uc[i];
   }
   
   Send(cmd_hex);
   return cmd_hex;
}

void myETL::SetCurrent(double current)
{
   this->LogMessage("SetCurrent function");
   Purge();
   CreateAndSendCurrentCommand(current);
   Purge();
}

void myETL::GetCurrent(double& current)
{
   this->LogMessage("GetCurrent function");

   // initialisation / declaration
   int i1 = 0, i2 = 0;
   int nLoop = 0, nRet = 0;
   unsigned long buflen = 6;
   unsigned long bytesRead = 0;
   unsigned long bytesToRead = 6;
   unsigned char* buf = new unsigned char[buflen];
   unsigned long totalBytes;
   totalBytes = 0;
   buf[0] = 0;

   Purge();

   // Get command
   unsigned char getCurrentCmd[6] = { 0x41, 0x72, 0x00, 0x00, 0xb4, 0x27 };
   Send(getCurrentCmd);
   
   // read
   while (nRet == 0 && totalBytes < bytesToRead && nLoop<100)
   {
      nRet = ReadFromComPort(port_.c_str(), &buf[totalBytes], buflen - totalBytes, bytesRead);
      nLoop++;
      totalBytes += bytesRead;
      CDeviceUtils::SleepMs(1);
   }

   // empirical conversion: I fixed the bugs one by one
   if (bytesRead == bytesToRead)
   {
      this->LogMessage("current read");
      this->LogMessage((char*)buf);
      if ((int)buf[1] > 127)
         i1 = -(256+(int)(~buf[1] + 1)); // 2's complement if negatif
      else
         i1 = (int)buf[1];
      
      if (i1>0)
         i2 = (int)buf[2];
      else
         i2 = (int)buf[2];
      
      current = i1*255+i2;
      current = current*293/4096; 
      this->LogMessage("current = " + std::to_string(current));
   }

   Purge();

}

/////////////////////////////////////
//  Communications
/////////////////////////////////////

void myETL::Send(unsigned char* cmd)
{
   unsigned int length = 6;
   int ret = WriteToComPort(port_.c_str(), cmd, length);
   if (ret != DEVICE_OK)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;
}

void myETL::Send(char* cmd)
{
   /*int ret = SendSerialCommand(port_.c_str(), cmd, end_of_line);
   if (ret != DEVICE_OK)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;*/
}

void myETL::Send(std::string cmd)
{
   /*int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), end_of_line); // carriage_return before
   if (ret != DEVICE_OK)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;*/
}

void myETL::Purge()
{
   int ret = PurgeComPort(port_.c_str());
   if (ret != 0)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;
}


// Communication "clear buffer" utility function:
/*int myETL::ClearPort(void)
{
   // Clear contents of serial port
   const int bufSize = 255;
   unsigned char clear[bufSize];
   unsigned long read = bufSize;
   int ret;
   while ((int)read == bufSize)
   {
      ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufSize, read);
      if (ret != DEVICE_OK)
         return ret;
   }
   return DEVICE_OK;
}*/

// Communication "send" utility function:
/*int myETL::SendCommand(const char* command) const
{
   const char* g_TxTerm = "\r";
   int ret;

   std::string base_command = "";
   base_command += command;
   // send command
   ret = core_->SetSerialCommand(device_, port_.c_str(), base_command.c_str(), g_TxTerm);
   return ret;
}*/

// Communication "send & receive" utility function:
/*int myETL::QueryCommand(const char* command, std::string& answer) const
{
   const char* g_RxTerm = "\r";
   int ret;

   // send command
   ret = SendCommand(command);

   if (ret != DEVICE_OK)
      return ret;
   // block/wait for acknowledge (or until we time out)
   const size_t BUFSIZE = 2048;
   char buf[BUFSIZE] = { '\0' };
   ret = core_->GetSerialAnswer(device_, port_.c_str(), BUFSIZE, buf, g_RxTerm);
   answer = buf;
   return ret;
}*/

/*int myETL::CheckDeviceStatus(void)
{
   int ret = ClearPort();
   if (ret != DEVICE_OK) return ret;

   // LStep Version
   std::string resp;
   ret = QueryCommand("?ver", resp);
   if (ret != DEVICE_OK) return ret;
   if (resp.length() < 1) return  DEVICE_NOT_CONNECTED;
   //expected response starts either with "Vers:LS" or "Vers:LP"
   if (resp.find("Vers:L") == std::string::npos) return DEVICE_NOT_CONNECTED;


   ret = SendCommand("!autostatus 0"); //diasable autostatus
   if (ret != DEVICE_OK) return ret;

   ret = QueryCommand("?det", resp);
   if (ret != DEVICE_OK) return ret;
   if (resp.length() < 1) return DEVICE_SERIAL_INVALID_RESPONSE;
   Configuration_ = atoi(resp.c_str());

   initialized_ = true;
   return DEVICE_OK;
}*/