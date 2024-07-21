#include "squid.h"


/*
 * Utility class for SquidMonitoringThread
 * Takes an input stream and returns CAN29 messages in the GetNextMessage method
 */
SquidMessageParser::SquidMessageParser(unsigned char* inputStream, long inputStreamLength) :
   index_(0)
{
   inputStream_ = inputStream;
   inputStreamLength_ = inputStreamLength;
}

/*
 */
int SquidMessageParser::GetNextMessage(unsigned char* nextMessage, int& nextMessageLength) {
   nextMessageLength = 0;
   while ((index_ < inputStreamLength_) && (nextMessageLength < messageMaxLength_) ) {
      nextMessage[nextMessageLength] = inputStream_[index_];
      nextMessageLength++;
      index_++;
   }
   if (nextMessageLength == messageMaxLength_)
      return 0;
   else {
      return -1;
   }
}


SquidMonitoringThread::SquidMonitoringThread(MM::Core& core, SquidHub& hub, bool debug) :
   core_(core),
   hub_(hub),
   debug_(debug),
   stop_(true),
   intervalUs_(10000), // check every 10 ms for new messages, 
   ourThread_(0)
{
   //deviceInfo = deviceInfo_;
}

SquidMonitoringThread::~SquidMonitoringThread()
{
   stop_ = true;
   ourThread_->join();
   //hub_.LogMessage("Destructing MonitoringThread", true);
}

void SquidMonitoringThread::interpretMessage(unsigned char* message)
{
   if (message[1] != 0x0) {
      if (debug_) {
         std::ostringstream os;
         os << "Monitoring Thread incoming message: ";
         for (int i = 0; i < SquidMessageParser::messageMaxLength_; i++) {
            os << std::hex << (unsigned int)message[i] << " ";
         }
         core_.LogMessage(&hub_, os.str().c_str(), false);
      }
   }

}


int SquidMonitoringThread::svc() {

   //core_.LogMessage(&device_, "Starting MonitoringThread", true);

   unsigned long dataLength;
   unsigned long charsRead = 0;
   unsigned long charsRemaining = 0;
   unsigned char rcvBuf[SquidHub::RCV_BUF_LENGTH];
   memset(rcvBuf, 0, SquidHub::RCV_BUF_LENGTH);

   while (!stop_)
   {
      do {
         if (charsRemaining > (SquidHub::RCV_BUF_LENGTH - SquidMessageParser::messageMaxLength_)) {
            // for one reason or another, our buffer is overflowing.  Empty it out before we crash
            charsRemaining = 0;
         }
         dataLength = SquidHub::RCV_BUF_LENGTH - charsRemaining;

         // Do the scope monitoring stuff here
         // MM::MMTime _start = core_.GetCurrentMMTime();

         int ret = core_.ReadFromSerial(&hub_, hub_.port_.c_str(), rcvBuf + charsRemaining, dataLength, charsRead);

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
               unsigned char message[SquidHub::RCV_BUF_LENGTH];
               int messageLength;
               ret = parser.GetNextMessage(message, messageLength);
               if (ret == 0) {
                  // Report 
                  // and do the real stuff
                  interpretMessage(message);
               }
               else {
                  // no more messages, copy remaining (if any) back to beginning of buffer
                  if (debug_ && messageLength > 0) {
                     std::ostringstream os;
                     os << "Monitoring Thread no message found!: ";
                     for (int i = 0; i < messageLength; i++) {
                        os << std::hex << (unsigned int)message[i] << " ";
                        rcvBuf[i] = message[i];
                     }
                     //core_.LogMessage(&hub_, os.str().c_str(), false);
                  }
                  memset(rcvBuf, 0, SquidHub::RCV_BUF_LENGTH);
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
