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

#include "ScientificaRxPacket.h"

#define MAX_PACKET_SIZE 256

ScientificaRxPacket::ScientificaRxPacket(unsigned char* data, int data_length)
{
	index_ = 0;
	data_ = new unsigned char[MAX_PACKET_SIZE];

	//COBS Decoding
	int block_remaining = 0;
	int block_length = 0xFF;
	int write = 0;

	for (int i = 0; i < MAX_PACKET_SIZE; i++)
	{
		data_[i] = 0;
	}

	for (int i = 0; i < data_length; i++)
	{
		if (block_remaining != 0)
		{
			if (data[i] != 0)
			{
				data_[write] = data[i];
				write++;
			}
		}
		else
		{
			if (block_length != 0xFF)
			{
				data_[write] = 0;
				write++;
			}
			block_remaining = data[i];
			block_length = data[i];
		}

		block_remaining--;
	}

	data_length_ = write;

	if (data_length_ < 4)
	{
		status = 0;
		echo = 0;
		command1 = 0;
		command2 = 0;
	}
	else
	{
		status = data_[0];
		echo = data_[1];
		command1 = data_[2];
		command2 = data_[3];
		index_ = 4;
	}
}

ScientificaRxPacket::~ScientificaRxPacket()
{
	delete[] data_;
}

bool ScientificaRxPacket::GetByte(uint8_t* value)
{
	if (index_ + 1 > data_length_)
		return false;

	*value = data_[index_];
	index_++;
	return true;
}

bool ScientificaRxPacket::GetUInt16(uint16_t* value)
{
	if (index_ + 2 > data_length_)
		return false;

	*value = data_[index_];
	*value |= data_[index_ + 1] << 8;
	index_ += 2;
	return true;
}

bool ScientificaRxPacket::GetInt32(int32_t* value)
{
	if (index_ + 4 > data_length_)
		return false;

	*value = data_[index_];
	*value |= data_[index_ + 1] << 8;
	*value |= data_[index_ + 2] << 16;
	*value |= data_[index_ + 3] << 24;
	index_ += 4;
	return true;
}