/*
   简单算子
*/
#include "XTDef.h"
#include <string>

INT64 StrHexToDec(CString szStr)
{
	INT64 nTemp = 0;
	INT64 nData = 0;
	int nLen = szStr.GetLength();
	for (int i = 0; i<nLen; i++)
	{
		TCHAR c = szStr.GetAt(i);
		if (c >= '0' && c <= '9')
		{
			nTemp = c - '0';
		}
		else if (c >= 'a' && c <= 'f')
		{
			nTemp = c - 'a' + 10;
		}
		else if (c >= 'A' && c <= 'F')
		{
			nTemp = c - 'A' + 10;
		}
		else
		{
			return FALSE;
		}
		nTemp <<= ((nLen - i - 1) << 2);
		nData |= nTemp;
	}
	return nData;
}

// 将 IP 地址转换为整数
INT64 IPToInt(const char *IP)
{
	INT64 ip = 0;
	CString str = CString(IP);
	std::string strPos = str.GetBuffer();
	std::string tempPos;
	int nPos = 0, nTmp = 0, i = 0;
	while (strPos.length() != 0)
	{
		nPos = strPos.find('.', 0);
		tempPos = strPos.substr(0, nPos);
		nTmp = atoi(tempPos.c_str());
		ip = (ip << 8) | nTmp;
		strPos = strPos.substr(nPos + 1);
		i++;
		if (i >= 4)
		{
			break;
		}
	}
	return ip;
}

// 将整数转换为 IP 地址
CString IntToIP(const INT64 value)
{
	CString txt;
	txt.Format("%d.%d.%d.%d", (value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, (value)& 0xFF);
	return txt;
}

// 将 MAC 地址转换为整数
INT64 MacToInt(const char *mac)
{
	char mac_str[18];
	strcpy(mac_str, mac);
	char *ptr = strtok(mac_str, ":");
	INT64 mac_int = 0;
	while (ptr != NULL) {
		mac_int = (mac_int << 8) + strtoul(ptr, NULL, 16);
		ptr = strtok(NULL, ":");
	}
	return mac_int;
}

// 将整数转换为 MAC 地址
CString IntToMac(const INT64 value)
{
	CString txt;
	txt.Format("%02llX:%02llX:%02llX:%02llX:%02llX:%02llX",
		(value >> 40) & 0xFF, (value >> 32) & 0xFF, (value >> 24) & 0xFF,
		(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
	return txt;
}
