#include "Squid.h"
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24));


/*
 * Utility class for SquidMonitoringThread
 * Takes an input stream and returns messages in the GetNextMessage method
 */
SquidMessageParser::SquidMessageParser(unsigned char* inputStream, long inputStreamLength) :
   index_(0)
{
   inputStream_ = inputStream;
   inputStreamLength_ = inputStreamLength;
}

/*
 * Provides the next message in the inputStream.
 * Returns 0 on success, -1 when no message was found
 */
int SquidMessageParser::GetNextMessage(unsigned char* nextMessage, int& nextMessageLength) {

   bool msgFound = false;
   while (!msgFound)
   {
      nextMessageLength = 0;
      while ((index_ < inputStreamLength_) && (nextMessageLength < messageMaxLength_)) {
         nextMessage[nextMessageLength] = inputStream_[index_];
         nextMessageLength++;
         index_++;
      }

      if (nextMessageLength == messageMaxLength_)
      {
         msgFound = true;
         return 0;
      }
      else //  index_ == inputStreamLength_
      {
         return -1;
      }
   }
   return -1; // should never be reached
}


SquidMonitoringThread::SquidMonitoringThread(MM::Core& core, SquidHub& hub, bool debug) :
   core_(core),
   hub_(hub),
   debug_(debug),
   stop_(true),
   intervalUs_(10000), // check every 10 ms for new messages, 
   ourThread_(0),
   counter_(0)
{
   isBigEndian_ = IsBigEndian();
}

SquidMonitoringThread::~SquidMonitoringThread()
{
   stop_ = true;
   ourThread_->join();
   //hub_.LogMessage("Destructing MonitoringThread", true);
}

/*
0 - command id of the last received command
1 - either COMMAND_CHECKSUM_ERROR(2), IN_PROGRESS(1), or COMPLETED_WITHOUT_ERRORS(0)
2 - 5 - uint32_t x pos in Big Endian
6 - 9 - uint32_t y pos in Big Endian
10 - 13 - uint32_t Z pos in Big Endian
14 - 17 - uint32_t Theta position
18 - Joystick Button
19 - 22 - Reserved
23 - CRC checksum(appears not to be set)
*/

void SquidMonitoringThread::InterpretMessage(unsigned char* message)
{
   if (debug_ && (counter_ % 100 == 0)) {
      std::ostringstream os;
      os << "Monitoring Thread incoming message: ";
      for (int i = 0; i < SquidMessageParser::messageMaxLength_; i++) {
         os << std::hex << (unsigned int)message[i] << " ";
      }
      core_.LogMessage(&hub_, os.str().c_str(), true);
   }
   counter_++;

   hub_.SetCmdNrReceived(message[0], message[1]);

   std::uint32_t ux;
   memcpy(&ux, &message[2], 4);
   if (!isBigEndian_)
   {
      ux = SWAP_INT32(ux);
   }
   hub_.SetPositionXSteps(ux);
   std::uint32_t uy;
   memcpy(&uy, &message[6], 4);
   if (!isBigEndian_)
   {
      uy = SWAP_INT32(uy);
   }
   hub_.SetPositionYSteps(uy);
   std::uint32_t uz;
   memcpy(&uz, &message[10], 4);
   if (!isBigEndian_)
   {
      uz = SWAP_INT32(uz);
   }
   hub_.SetPositionZSteps(uz);

}


int SquidMonitoringThread::svc() {

   //core_.LogMessage(&device_, "Starting MonitoringThread", true);

   unsigned long dataLength;
   unsigned long charsRead = 0;
   unsigned long charsRemaining = 0;
   unsigned char rcvBuf[RCV_BUF_LENGTH];
   memset(rcvBuf, 0, RCV_BUF_LENGTH);

   while (!stop_)
   {
      do {
         if (charsRemaining > (RCV_BUF_LENGTH - SquidMessageParser::messageMaxLength_)) {
            // for one reason or another, our buffer is overflowing.  Empty it out before we crash
            charsRemaining = 0;
         }
         dataLength = RCV_BUF_LENGTH - charsRemaining;

         // Do the scope monitoring stuff here
         // MM::MMTime _start = core_.GetCurrentMMTime();

         int ret = core_.ReadFromSerial(&hub_, hub_.port_.c_str(), rcvBuf + charsRemaining, dataLength, charsRead);

         //std::ostringstream os;
         //os << "Monitoring Thread read from Serial: ";
         //for (int i = 0; i < charsRead; i++) {
         //   os << std::hex << (unsigned int)rcvBuf[i] << " ";
         //}
         //core_.LogMessage(&hub_, os.str().c_str(), false);
  
         // MM::MMTime _end = core_.GetCurrentMMTime();
         // std::ostringstream os;
         // MM::MMTime t = _end-_start;
         // os << "ReadFromSerial took: " << t.sec_ << " seconds and " << t.uSec_ / 1000.0 << " msec";
         // core_.LogMessage(&device_, os.str().c_str(), true);

         if (ret != DEVICE_OK) {
            std::ostringstream oss;
            oss << "Monitoring Thread: ERROR while reading from serial port, error code: " << ret;
            core_.LogMessage(&hub_, oss.str().c_str(), false);
         }
         else if (charsRead > 0) {
            SquidMessageParser parser(rcvBuf, charsRead + charsRemaining);
            do {
               unsigned char message[RCV_BUF_LENGTH];
               int messageLength;
               ret = parser.GetNextMessage(message, messageLength);
               if (ret == 0) {
                  // Report 
                  // and do the real stuff
                  InterpretMessage(message);
               }
               else {
                  // no more messages, copy remaining (if any) back to beginning of buffer
                  if (debug_ && messageLength > 0) {
                     std::ostringstream oos;
                     oos << "Monitoring Thread no message found!: ";
                     for (int i = 0; i < messageLength; i++) {
                        oos << std::hex << (unsigned int)message[i] << " ";
                        rcvBuf[i] = message[i];
                     }
                     //core_.LogMessage(&hub_, os.str().c_str(), false);
                  }
                  memset(rcvBuf, 0, RCV_BUF_LENGTH);
                  for (int i = 0; i < messageLength; i++) {
                     rcvBuf[i] = message[i];
                  }
                  charsRemaining = messageLength;
               }
            } while (ret == 0);
         }
      } while ((charsRead != 0) && (!stop_));
      CDeviceUtils::SleepMs(intervalUs_ / 1000);
   }
    return 0;
}


void SquidMonitoringThread::Start()
{
   stop_ = false;
   ourThread_ = new std::thread(&SquidMonitoringThread::svc, this);
}

bool SquidMonitoringThread::IsBigEndian(void)
{
   union {
      uint32_t i;
      char c[4];
   } bint = { 0x01020304 };

   return bint.c[0] == 1;
}
