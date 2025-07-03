#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

class SingleThread {
public:
   SingleThread();
   ~SingleThread();
   void enqueue(std::function<void()> task);

private:
   void run(); 

   std::thread thread;
   std::queue<std::function<void()>> tasks;
   std::mutex mutex;
   std::condition_variable condition;
   bool running;
};

