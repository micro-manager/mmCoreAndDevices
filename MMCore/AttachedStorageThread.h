#pragma once
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

class CMMCore;

class AttachedStorageThread {
private:
   std::thread thread_;
   std::atomic<bool> running_;
   std::atomic<bool> shouldStop_;
   std::mutex mutex_;
   std::condition_variable cv_;
   CMMCore* core_;
   std::string errorMessage_;

   void workFunction();

public:
   AttachedStorageThread(CMMCore* core) : running_(false), shouldStop_(false), core_(core) {}
   ~AttachedStorageThread();

   void start();
   void stop();
   bool isRunning() const;
   std::string getErrorMessage() { return errorMessage_; }
};
