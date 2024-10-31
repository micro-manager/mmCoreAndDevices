///////////////////////////////////////////////////////////////////////////////
// FILE:          ScientificaRxPacket.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Helper class to parse packets from Scientifica motion 8 racks
//                 
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//                
// AUTHOR:        Matthew Player (ElecSoft Solutions)

#pragma once
#include <cstdint>
class ScientificaRxPacket
{
public:
	/**
	* @brief Constructor
	*/
	ScientificaRxPacket(unsigned char* buffer, int buffer_size);

	/**
	 * @brief Destructor
	 */
	~ScientificaRxPacket();

	/**
	 * @brief Get next unsigned 8 bit integer from packet
	 * @param[in] value  Destination for value
	 * @return true if successful, false if not enough data
	 */
	bool GetByte(uint8_t* value);

	/**
	* @brief Get next unsigned 16 bit integer from packet
	* @param[in] value  Destination for value
	* @return true if successful, false if not enough data
	*/	
	bool GetUInt16(uint16_t* value);

	/**
	* @brief Get next signed 32 bit integer from packet
	* @param[in] value  Destination for value
	* @return true if successful, false if not enough data
	*/
	bool GetInt32(int32_t* value);

	/**
	* @brief Get the remaining number of bytes in the packet
	* @return The number of bytes remaining in the packet
	*/
	int RemainingBytes() { return data_length_ - index_; }

	/**
	* @brief Get the length of the packet
	* @return The length of the packet
	*/
	int Length() { return data_length_; }

	/**
	* @brief Get the data buffer
	* @return The data buffer
	*/
	unsigned char* GetData() { return data_; }

	/**
	* @brief Skip the next given number of bytes in the packet
	* @param[in] count  The number of bytes to skip
	*/
	void Skip(int count) { index_ += count; }
	

private:
	unsigned char* data_;
	int data_length_;
	int index_;

	uint8_t status;
	uint8_t echo;
	uint8_t command1;
	uint8_t command2;
};

