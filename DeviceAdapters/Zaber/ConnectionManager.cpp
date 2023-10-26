#include "ConnectionManager.h"

std::shared_ptr<zml::Connection> ConnectionManager::getConnection(std::string port)
{	
	std::lock_guard<std::mutex> lockGuard(lock_);
	if (connections_.count(port) > 0) {
		if (auto connectionPtr = connections_.at(port).lock()) {
			return connectionPtr;
		}
	}

	auto connection = std::make_shared<zml::Connection>(zml::Connection::openSerialPort(port));
	auto id = connection->getInterfaceId();
	connection->getDisconnected().subscribe([=](std::shared_ptr<zmlbase::MotionLibException>) {
		removeConnection(port, id);
	});
	connections_[port] = connection;

	return connection;
}

bool ConnectionManager::removeConnection(std::string port, int interfaceId)
{
	std::lock_guard<std::mutex> lockGuard(lock_);
	if (connections_.count(port) == 0) {
		return false;
	} 

	if (interfaceId != -1) {
		if (auto connection = connections_.at(port).lock()) {
			if (connection->getInterfaceId() != interfaceId) {
				return false;
			}
		}
	}

	connections_.erase(port);
	return true;
}