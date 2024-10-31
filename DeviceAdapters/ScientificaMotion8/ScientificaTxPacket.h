///////////////////////////////////////////////////////////////////////////////
// FILE:          ScientificaTxPacket.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Helper class to build packets to send to Scientifica motion 8 racks
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

class ScientificaTxPacket
{
public: 
	/**
	* @brief Constructor
	*/
	ScientificaTxPacket(uint8_t echo, uint8_t command1, uint8_t command2);

	/**
	 * @brief Destructor
	 */
	~ScientificaTxPacket();

	/**
	 * @brief Get the encoded packet ready to send
	 * @return encoded packet to send
	 */
	unsigned char* GetPacketToSend();

	/**
	* @brief Get the length of the encoded packet
	* @return The length of the encoded packet
	*/
	int GetEncodedLength() { return encoded_length_; }

	/**
	* @brief Clear all data from the packet
	*/
	void Clear();
	
	/**
	 * @brief Add an unsigned 8 bit integer to packet
	 * @param[in] byte value to add
	 */
	void AddUInt8(uint8_t byte);

	/**
	 * @brief Add an unsigned 16 bit integer to packet
	 * @param[in] word value to add
	 */
	void AddUInt16(uint16_t word);

	/**
	 * @brief Add an unsigned 32 bit integer to packet
	 * @param[in] dword value to add
	 */
	void AddUInt32(uint32_t dword);

	/**
	 * @brief Add an signed 32 bit integer to packet
	 * @param[in] dword value to add
	 */
	void AddInt32(int32_t dword);

private:
	uint8_t* packet_;
	uint8_t* encoded_;
	int write_index_;
	int encoded_length_;
};

