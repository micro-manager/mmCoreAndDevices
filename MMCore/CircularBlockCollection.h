///////////////////////////////////////////////////////////////////////////////
// FILE:          CircularBlockCollection.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implementation of circular buffer that also stores sizes
//				  of contained data blocks
//              
// COPYRIGHT:     Artem Melnykov, 2022
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
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
// AUTHOR:        Artem Melnykov, melnykov.artem@gmail.com, 2022
// 

#pragma once

#include <boost/circular_buffer.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#include "../MMDevice/DeviceThreads.h"

typedef struct {
	boost::shared_ptr<char*> data;
	unsigned int dataSize;
} BLOCK;


class CircularBlockCollection
{
public:
	CircularBlockCollection(unsigned int maxNumberOfBlocks, bool stopOnOverflow);
	~CircularBlockCollection();

	bool GetOverflowStatus();
	void ResetOverflowStatus();

	void Add(BLOCK dataBlock);
	BLOCK Remove();

private:
	boost::circular_buffer<BLOCK> cb_;
	MMThreadLock executeLock_;
	unsigned long maxNumberOfBlocks_;
	bool stopOnOverflow_;
	bool overflowStatus_; 
};
