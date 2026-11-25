#ifndef _TX_PRINGT_H_
#define _TX_PRINGT_H_
#include <stdio.h>
#include <map>
#include <atlstr.h>
using std::map;
using std::pair;

#define TUDBG_PRINTF(format,...)	{char dbgMsg[1024] = {0}; sprintf_s(dbgMsg, "" format"", ##__VA_ARGS__); OutputDebugString(dbgMsg);}

#define CAMPID_GIGE         0x0005      // GIGE Camera
#define CAMPID_CXP          0x0007      // CXP Camera

typedef map<char *, void *>  MAP_TUELEMENT;

INT64   StrHexToDec(CString szStr);
INT64   IPToInt(const char *IP);
CString IntToIP(const INT64 value);
INT64   MacToInt(const char *mac);
CString IntToMac(const INT64 value);

#endif  // _TX_PRINGT_H_