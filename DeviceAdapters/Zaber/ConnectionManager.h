#pragma once

#include <MMDevice.h>
#include <zaber/motion/ascii/connection.h>
#include <map>
#include <string>
#include <mutex>
#include <memory>

namespace zmlbase = zaber::motion;
namespace zml = zaber::motion::ascii;

class ConnectionManager
{
public:
	std::shared_ptr<zml::Connection> getConnection(std::string port);
	bool removeConnection(std::string port, int interfaceId = -1);
private:
	std::mutex lock_;
	std::map<std::string, std::weak_ptr<zml::Connection>> connections_;
};

