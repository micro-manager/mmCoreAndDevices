///////////////////////////////////////////////////////////////////////////////
// FILE:          CircularBlockCollection.cpp
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

#include "CircularBlockCollection.h"


CircularBlockCollection::CircularBlockCollection(unsigned int maxNumberOfBlocks, bool stopOnOverflow) :
	maxNumberOfBlocks_(0),
	stopOnOverflow_(true),
	overflowStatus_(false)
{
	boost::circular_buffer<BLOCK> cb_(maxNumberOfBlocks);
	maxNumberOfBlocks_ = maxNumberOfBlocks;
    stopOnOverflow_ = stopOnOverflow;
}

CircularBlockCollection::~CircularBlockCollection()
{
}

void CircularBlockCollection::Add(BLOCK dataBlock)
{
	MMThreadGuard g(this->executeLock_);
	if (cb_.capacity() - cb_.size() == 0) {
		overflowStatus_ = true;
		if (stopOnOverflow_) return;
		cb_.empty();
	}
	cb_.push_front(dataBlock);
}

BLOCK CircularBlockCollection::Remove()
{
	MMThreadGuard g(this->executeLock_);
	if (cb_.size() == 0) {
		return BLOCK();
	}
	else {
		BLOCK dataBlock = BLOCK(cb_.at(0));
		cb_.pop_back();
		return dataBlock;
	}
}

void CircularBlockCollection::ResetOverflowStatus()
{
	MMThreadGuard g(this->executeLock_);
	overflowStatus_ = false;
}

bool CircularBlockCollection::GetOverflowStatus()
{
	MMThreadGuard g(this->executeLock_);
	return overflowStatus_;
}