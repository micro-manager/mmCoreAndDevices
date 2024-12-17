#include "TeensyCom.h"


TeensyCom::TeensyCom(MM::Core* callback, MM::Device* device, const char* portLabel) :
   callback_(callback),
   device_(device),
   portLabel_(portLabel)
{
   callback->PurgeSerial(device, portLabel_);
}

int TeensyCom::SendCommand(uint8_t cmd, uint32_t param)
{
   // This should not be needed, but just in case:
   
   callback_->PurgeSerial(device_, portLabel_);

    // Prepare buffer
   const unsigned int buflen = 5;
   unsigned char buffer[buflen];
   buffer[0] = cmd;
   // This needs work when we have a Big Endian host, the Teensy is little endian
   alignas(uint32_t) char alignedByteArray[4];
   std::memcpy(alignedByteArray, &param, 4);

   // For commands that require a parameter, convert to little-endian
   buffer[1] = alignedByteArray[0];
   buffer[2] = alignedByteArray[1];
   buffer[3] = alignedByteArray[2];
   buffer[4] = alignedByteArray[3];

   // Send 5-byte command
   return callback_->WriteToSerial(device_, portLabel_, buffer, buflen);
}

int TeensyCom::Enquire(uint8_t cmd)
{
   const unsigned int buflen = 2;
   unsigned char buffer[buflen];
   buffer[0] = 255;
   buffer[1] = cmd;

   return callback_->WriteToSerial(device_, portLabel_, buffer, buflen);
}


int TeensyCom::GetResponse(uint8_t cmd, uint32_t& param)
{
   unsigned char buf[1] = { 0 };
   unsigned long read = 0;
   MM::TimeoutMs timeout = MM::TimeoutMs(callback_->GetCurrentMMTime(), 1000);
   while (read == 0) {
      if (timeout.expired(callback_->GetCurrentMMTime()))
      {
         return ERR_COMMUNICATION;
      }
      callback_->ReadFromSerial(device_, portLabel_, buf, 1, read);
   }
   if (read == 1 && buf[0] == cmd)
   {
      unsigned char buf2[4] = { 0, 0, 0, 0 };
      read = 0;
      unsigned long tmpRead = 0;
      while (read < 4)
      {
         if (timeout.expired(callback_->GetCurrentMMTime()))
         {
            return ERR_COMMUNICATION;
         }
         callback_->ReadFromSerial(device_, portLabel_, &buf2[read], 4 - read, tmpRead);
         read += tmpRead;
      }
      alignas(uint32_t) char alignedByteArray[4];
      std::memcpy(alignedByteArray, buf2, sizeof(buf2));

      // This needs change on a Big endian host (PC and Teensy are both little endian).
      param = *(reinterpret_cast<uint32_t*>(alignedByteArray));

      std::ostringstream os;
      os << "Read: " << param;
      callback_->LogMessage(device_, os.str().c_str(), true);
   }
   else
   {
      return ERR_COMMUNICATION;
   }
   return DEVICE_OK;
}

int TeensyCom::GetRunningStatus(uint32_t& status)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = Enquire(cmd_start);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_start, status);
}

// get firmware version
int TeensyCom::GetVersion(uint32_t& version)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = SendCommand(cmd_version, 0);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(0, version);
}

// Get interval between pulses
int TeensyCom::GetInterval(uint32_t& interval)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = Enquire(cmd_interval);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_interval, interval);
}

int TeensyCom::GetPulseDuration(uint32_t& pulseDuration)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = Enquire(cmd_pulse_duration);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_pulse_duration, pulseDuration);
}

int TeensyCom::GetWaitForInput(uint32_t& waitForInput)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = Enquire(cmd_wait_for_input);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_wait_for_input, waitForInput);
}

int TeensyCom::GetNumberOfPulses(uint32_t& nrPulses)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = Enquire(cmd_number_of_pulses);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_number_of_pulses, nrPulses);
}

int TeensyCom::SetStart(uint32_t& response)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = SendCommand(cmd_start, 0); // Start command
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_start, response);
}

int TeensyCom::SetStop(uint32_t& response)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = SendCommand(cmd_stop, 0); // Start command
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_stop, response);
}

int TeensyCom::SetInterval(uint32_t interval, uint32_t& response)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    int ret = SendCommand(cmd_interval, interval);
    if (ret != DEVICE_OK)
       return ret;
    return GetResponse(cmd_interval, response);
}


// Sets the pulse duration in microseconds
// return error on serial comm problem.  
// response contains Teensy reposonse (hopefully same as pulseDuration).
int TeensyCom::SetPulseDuration(uint32_t pulseDuration, uint32_t& response)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    int ret = SendCommand(cmd_pulse_duration, pulseDuration);
    if (ret != DEVICE_OK)
       return ret;
    return GetResponse(cmd_pulse_duration, response);
}


int TeensyCom::SetWaitForInput(uint32_t waitForInput, uint32_t& response)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = SendCommand(cmd_wait_for_input, waitForInput);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_wait_for_input, response);
}


int TeensyCom::SetNumberOfPulses(uint32_t numberOfPulses, uint32_t& response)
{
   const std::lock_guard<std::mutex> lock(mutex_);
   int ret = SendCommand(cmd_number_of_pulses, numberOfPulses);
   if (ret != DEVICE_OK)
      return ret;
   return GetResponse(cmd_number_of_pulses, response);
}


