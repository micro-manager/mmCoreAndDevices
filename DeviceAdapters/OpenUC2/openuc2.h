#ifndef _OPENUC2_H_
#define _OPENUC2_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>

// Global device names
static const char* g_HubName       = "openUC2-Hub";
static const char* g_XYStageName   = "openUC2-XYStage";
static const char* g_ZStageName    = "openUC2-ZStage";
static const char* g_ShutterName   = "openUC2-LED-Laser";

// Example error code
enum {
   ERR_NO_PORT_SET = 1001
};

#endif // _OPENUC2_H_
