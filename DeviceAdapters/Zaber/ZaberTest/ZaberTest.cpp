#include "../../../MMCore/MMCore.h"
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <sstream>

using namespace std;

int main(int argc, char* argv[])
{
   // Get module and device names
   if (argc < 4)
   {
      cout << "Error. Module, device, or port name not specified!" << endl;
      cout << "ZaberTest <module_name> <device_name> <port_name>" << endl;
      return 1;
   }
   else if (argc > 4)
   {
      cout << "Error. Too many parameters!" << endl;
      cout << "ZaberTest <module_name> <device_name> <port_name>" << endl;
      return 1;
   }
   string moduleName(argv[1]);
   string deviceName(argv[2]);
   string portName(argv[3]);

   CMMCore core;
   core.enableStderrLog(true);
   core.enableDebugLog(true);
   string label("Device");
   try
   {
      // Initialize the device
      // ---------------------

      cout << "Loading " << deviceName << " from library " << moduleName << "..." << endl;
      core.loadDevice(label.c_str(), moduleName.c_str(), deviceName.c_str());
      cout << "Done." << endl;

      core.setProperty(label.c_str(), "Zaber Serial Port", portName.c_str());

      cout << "Initializing..." << endl;
      core.initializeAllDevices();
      cout << "Done." << endl;

      // Obtain device properties
      // ------------------------
      vector<string> props(core.getDevicePropertyNames(label.c_str()));
      for (unsigned i=0; i < props.size(); i++)
      {
         cout << props[i] << " (" << core.getPropertyType(label.c_str(), props[i].c_str()) << ") = "
                          << core.getProperty(label.c_str(), props[i].c_str()) << endl;
      }

      // unload the device
      // -----------------
      core.unloadAllDevices();
   }
   catch (CMMError& err)
   {
      cout << err.getMsg();
      return 1;
   }

   // declare success
   // ---------------
   cout << "Device " + deviceName + " PASSED" << endl;
	return 0;
}
