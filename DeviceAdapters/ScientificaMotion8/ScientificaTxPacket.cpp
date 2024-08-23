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

#include "ScientificaTxPacket.h"

#define MAX_PACKET_SIZE 256

ScientificaTxPacket::ScientificaTxPacket(uint8_t echo, uint8_t command1, uint8_t command2)
{
	packet_ = new uint8_t[MAX_PACKET_SIZE];
	encoded_ = new uint8_t[MAX_PACKET_SIZE];
	write_index_ = 0;
    encoded_length_ = 0;

    packet_[0] = echo;
    packet_[1] = command1;
    packet_[2] = command2;
    write_index_ = 3;
}
ScientificaTxPacket::~ScientificaTxPacket()
{
	delete[] packet_;
	delete[] encoded_;
}

unsigned char* ScientificaTxPacket::GetPacketToSend()
{
    int block_start = 0;
    int block_length = 1;
    int write = 1;

    for (int i = 0; i < MAX_PACKET_SIZE; i++)
    {
        encoded_[i] = 0;
    }
    //COBS Encoding
    for (int i = 0; i < write_index_; i++)
    {
        if (packet_[i] != 0)
        {
            encoded_[write] = packet_[i];
            block_length++;
            write++;
        }

        if ((block_length == 0xFF) || (packet_[i] == 0))
        {
            encoded_[block_start] = block_length;
            block_start = write;
            encoded_[write] = 0;
            write++;
            block_length = 1;
        }
    }
    encoded_[block_start] = block_length;
    encoded_[write] = 0x0; //Add packet deliminator
    write++;

    encoded_length_ = write;
	return (unsigned char*)encoded_;
}

void ScientificaTxPacket::Clear()
{
	write_index_ = 0;
}

void ScientificaTxPacket::AddUInt8(uint8_t byte)
{
	packet_[write_index_++] = byte;
}

void ScientificaTxPacket::AddUInt16(uint16_t word)
{
	packet_[write_index_++] = word & 0xff;
	packet_[write_index_++] = (word >> 8) & 0xff;
}

void ScientificaTxPacket::AddUInt32(uint32_t dword)
{
	packet_[write_index_++] = dword & 0xff;
	packet_[write_index_++] = (dword >> 8) & 0xff;
	packet_[write_index_++] = (dword >> 16) & 0xff;
	packet_[write_index_++] = (dword >> 24) & 0xff;
}

void ScientificaTxPacket::AddInt32(int32_t dword)
{
	packet_[write_index_++] = dword & 0xff;
	packet_[write_index_++] = (dword >> 8) & 0xff;
	packet_[write_index_++] = (dword >> 16) & 0xff;
	packet_[write_index_++] = (dword >> 24) & 0xff;
}
