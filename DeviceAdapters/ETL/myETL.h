#ifndef _MYETL_H_
#define _MYETL_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceUtils.h"
#include <string>

#pragma once
class myETL : public CGenericBase<myETL>
{
public:
   // constructor - destructor
   myETL(const char* name);
   ~myETL();

   // MMDevice API
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // action interface
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCurrent(MM::PropertyBase* pProp, MM::ActionType eAct);

   // fonctions pour calculer la commande a envoyer en hexa
   void DecToHexa(int n, char* output);
   void StringToHexa(char* input, char* output);
   uint16_t BitReflect(uint16_t data, unsigned char nbits);
   uint16_t Crc16Ibm(char* data, unsigned int data_len);
   unsigned char* CreateAndSendCurrentCommand(double current);

   /*int ClearPort(void);
   int CheckDeviceStatus(void); // vient de LSTEP, ne correspond pas à ETL
   int SendCommand(const char* command) const;
   int QueryCommand(const char* command, std::string& answer) const;*/

private:
   bool initialized_;
   std::string port_;
   std::string name_;
   int error_;

   void GetCurrent(double& current);
   void SetCurrent(double current);
   void GeneratePropertyCurrent();
   void Purge();
   void Send(std::string cmd);
   void Send(char* cmd);
   void Send(unsigned char* cmd);
};

#endif //_MYETL_H_
