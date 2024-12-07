///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SFileUtil.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
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
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////
#include "G2SFileUtil.h"

/**
 * Write integer value to a byte buffer
 * @param buff Byte buffer
 * @param len Value length (in bytes)
 * @param val Integer value
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
void writeInt(unsigned char* buff, std::uint8_t len, std::uint64_t val) noexcept
{
	if(buff == nullptr || len == 0)
		return;
	for(auto i = 0; i < len; i++)
		buff[i] = (val >> (i * 8)) & 0xff;
}

/**
 * Read integer value from a byte buffer
 * @param buff Byte buffer
 * @param len Value length (in bytes)
 * @return Integer value
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
std::uint64_t readInt(const unsigned char* buff, std::uint8_t len) noexcept
{
	if(buff == nullptr || len == 0 || len > 8)
		return 0;
	std::uint64_t ret = 0;
	for(std::uint8_t i = 0; i < len; i++)
	{
		auto shift = i * 8;
		std::uint64_t xval = (std::uint64_t)buff[i] << shift;
		ret |= xval;
	}
	return ret;
}