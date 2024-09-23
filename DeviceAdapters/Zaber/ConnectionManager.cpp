#include "ConnectionManager.h"

#include <regex>
#include <zaber/motion/exceptions/invalid_argument_exception.h>

std::shared_ptr<zml::Connection> ConnectionManager::getConnection(std::string port)
{
	std::lock_guard<std::mutex> lockGuard(lock_);
	if (connections_.count(port) > 0) {
		if (auto connectionPtr = connections_.at(port).lock()) {
			return connectionPtr;
		}
	}

	std::shared_ptr<zml::Connection> connection;
	if (port.find("share://") == 0) {
		// share://<host>:<port>/<connection>
		std::regex parser("^share:\\/\\/([^:\\/]+)(:\\d+)?(\\/.*)?$", std::regex_constants::ECMAScript);
		std::smatch partMatch;
		if (!std::regex_match(port, partMatch, parser)) {
			throw zmlbase::InvalidArgumentException("Invalid network share connection string: " + port);
		}

		std::string host = partMatch[1].str();
		int sharePort = 11421;
		std::string connectionName;
		if (partMatch[2].matched) {
			sharePort = std::stoi(partMatch[2].str().substr(1));
		}
		if (partMatch[3].matched) {
			connectionName = partMatch[3].str().substr(1);
		}

		connection = std::make_shared<zml::Connection>(zml::Connection::openNetworkShare(host, sharePort, connectionName));
	} else {
		connection = std::make_shared<zml::Connection>(zml::Connection::openSerialPort(port));
	}

	auto id = connection->getInterfaceId();
	connection->getDisconnected().subscribe([=, this](std::shared_ptr<zmlbase::MotionLibException>) {
		removeConnection(port, id);
	});
	connections_[port] = connection;

	return connection;
}

bool ConnectionManager::removeConnection(std::string port, int interfaceId)
{
	std::lock_guard<std::mutex> lockGuard(lock_);
	auto it = connections_.find(port);
	if (it == connections_.end()) {
		return false;
	}

	if (interfaceId != -1) {
		if (auto connection = it->second.lock()) {
			if (connection->getInterfaceId() != interfaceId) {
				return false;
			}
		}
	}

	connections_.erase(it);
	return true;
}
