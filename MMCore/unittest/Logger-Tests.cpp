#include <gtest/gtest.h>

#include "Logging/Logging.h"

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace mm::logging;


TEST(LoggerTests, BasicSynchronous)
{
   std::shared_ptr<LoggingCore> c =
      std::make_shared<LoggingCore>();

   c->AddSink(std::make_shared<StdErrLogSink>(), SinkModeSynchronous);

   Logger lgr = c->NewLogger("mylabel");

   lgr(LogLevelDebug, "My entry text\nMy second line");
   for (unsigned i = 0; i < 1000; ++i)
      lgr(LogLevelDebug, "More lines!\n\n\n");
}


TEST(LoggerTests, BasicAsynchronous)
{
   std::shared_ptr<LoggingCore> c =
      std::make_shared<LoggingCore>();

   c->AddSink(std::make_shared<StdErrLogSink>(), SinkModeAsynchronous);

   Logger lgr = c->NewLogger("mylabel");

   lgr(LogLevelDebug, "My entry text\nMy second line");
   for (unsigned i = 0; i < 1000; ++i)
      lgr(LogLevelDebug, "More lines!\n\n\n");
}


TEST(LoggerTests, BasicLogStream)
{
   std::shared_ptr<LoggingCore> c =
      std::make_shared<LoggingCore>();

   c->AddSink(std::make_shared<StdErrLogSink>(), SinkModeSynchronous);

   Logger lgr = c->NewLogger("mylabel");

   LOG_INFO(lgr) << 123 << "ABC" << 456;
}


class LoggerTestThreadFunc
{
   unsigned n_;
   std::shared_ptr<LoggingCore> c_;

public:
   LoggerTestThreadFunc(unsigned n,
         std::shared_ptr<LoggingCore> c) :
      n_(n), c_(c)
   {}

   void Run()
   {
      Logger lgr =
         c_->NewLogger("thread" + std::to_string(n_));
      auto ch = '0' + n_;
      if (ch < '0' || ch > 'z')
         ch = '~';
      for (size_t j = 0; j < 50; ++j)
      {
         LOG_TRACE(lgr) << j << ' ' << std::string(n_ * j, char(ch));
      }
   }
};


TEST(LoggerTests, SyncAndThreaded)
{
   std::shared_ptr<LoggingCore> c =
      std::make_shared<LoggingCore>();

   c->AddSink(std::make_shared<StdErrLogSink>(), SinkModeSynchronous);

   std::vector< std::shared_ptr<std::thread> > threads;
   std::vector< std::shared_ptr<LoggerTestThreadFunc> > funcs;
   for (unsigned i = 0; i < 10; ++i)
   {
      funcs.push_back(std::make_shared<LoggerTestThreadFunc>(i, c));
      threads.push_back(std::make_shared<std::thread>(
               &LoggerTestThreadFunc::Run, funcs[i].get()));
   }
   for (unsigned i = 0; i < threads.size(); ++i)
      threads[i]->join();
}


TEST(LoggerTests, AsyncAndThreaded)
{
   std::shared_ptr<LoggingCore> c =
      std::make_shared<LoggingCore>();

   c->AddSink(std::make_shared<StdErrLogSink>(), SinkModeAsynchronous);

   std::vector< std::shared_ptr<std::thread> > threads;
   std::vector< std::shared_ptr<LoggerTestThreadFunc> > funcs;
   for (unsigned i = 0; i < 10; ++i)
   {
      funcs.push_back(std::make_shared<LoggerTestThreadFunc>(i, c));
      threads.push_back(std::make_shared<std::thread>(
               &LoggerTestThreadFunc::Run, funcs[i].get()));
   }
   for (unsigned i = 0; i < threads.size(); ++i)
      threads[i]->join();
}