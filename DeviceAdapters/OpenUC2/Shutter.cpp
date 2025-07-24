
#include "Shutter.h"
#include "UC2Hub.h"
#include <sstream>

UC2Shutter::UC2Shutter() :
   initialized_(false),
   hub_(nullptr),
   open_(false)
{
   // constructor
}

UC2Shutter::~UC2Shutter()
{
   Shutdown();
}

int UC2Shutter::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   hub_ = dynamic_cast<UC2Hub*>(GetParentHub());
   if (!hub_)
      return ERR_NO_PORT_SET;

   // e.g., create a property for LED or Laser power:
   // CreateIntegerProperty("Laser1-Power", 0, false, new CPropertyAction(this, &UC2Shutter::OnLaserPower));

   initialized_ = true;
   return DEVICE_OK;
}

int UC2Shutter::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

bool UC2Shutter::Busy()
{
   return false;
}

int UC2Shutter::SetOpen(bool open)
{
   if (!initialized_)
      return DEVICE_NOT_CONNECTED;

   // Example: if "open" means set laser or LED to 100%:
   int intensity = open ? 255 : 0;
   // JSON: /laser_act or /ledarr_act
   std::ostringstream ss;
   ss << R"({"task":"/laser_act", "LASERid":1,"LASERval":)"
      << intensity << "}";

   std::string cmd = ss.str();
   std::string reply;
   int ret = hub_->SendJsonCommand(cmd, reply);
   if (ret != DEVICE_OK)
      return ret;

   open_ = open;
   return DEVICE_OK;
}

int UC2Shutter::GetOpen(bool& open)
{
   // We cannot easily query the device, so we return our cached state:
   open = open_;
   return DEVICE_OK;
}
