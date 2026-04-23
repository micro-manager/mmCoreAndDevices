#include "PvCircularBuffer.h"

#include <cstddef> // ptrdiff_t

PvCircularBuffer::PvCircularBuffer()
{
}

PvCircularBuffer::~PvCircularBuffer()
{
}

int PvCircularBuffer::Capacity() const
{
    return frameCount_;
}

size_t PvCircularBuffer::FrameSize() const
{
    return frameSize_;
}


void* PvCircularBuffer::Data() const
{
    return pBuffer_.get();
}

size_t PvCircularBuffer::Size() const
{
    return size_;
}

int PvCircularBuffer::LatestFrameIndex() const
{
    return latestFrameIdx_;
}

void* PvCircularBuffer::FrameData(int index) const
{
    // No bounds checking. Most of the functions in this class are called very
    // often - with every frame which could be thousands of times per second so
    // we try to avoid unnecessary branching.
    return &pBuffer_[index * frameSize_];
}

const PvFrameInfo& PvCircularBuffer::FrameInfo(int index) const
{
    return pFrameInfoArray_[index];
}

void PvCircularBuffer::Reset()
{
    // Must be -1, once the very first frame arrives the index becomes 0
    latestFrameIdx_ = -1;
}

void PvCircularBuffer::Resize(size_t frameSize, int count)
{
    if (frameSize != frameSize_ || count != frameCount_)
    {
        frameCount_ = count;
        frameSize_  = frameSize;
        size_       = count * static_cast<size_t>(frameSize);
        pBuffer_.reset();
        // HACK! There still seems to be some heap corruption issues in PVCAM, in
        // debug builds I am getting heap corruption error on reset() here.
        // Adding just 16 bytes to the entire buffer seems to help.
        pBuffer_ = std::make_unique<unsigned char[]>(size_ + 16);

        pFrameInfoArray_.reset();
        pFrameInfoArray_ = std::make_unique<PvFrameInfo[]>(frameCount_);
    }

    Reset();
}

void PvCircularBuffer::ReportFrameArrived(const PvFrameInfo& frameNfo, void* pFrameData)
{
    // Calculate the index of the received frame in our circular buffer
    const int curFrameIdx = static_cast<const int>
        ((ptrdiff_t(pFrameData) - ptrdiff_t(pBuffer_.get())) / frameSize_);

    // Store a copy of the frameNfo to our array
    pFrameInfoArray_[curFrameIdx] = frameNfo;

    // In case we ever need that we can check missed frames in this place as well.
    // Currently we use FRAME_INFO.FrameNr to check for missed callbacks but here
    // we can use the current and last index to check whether they are in sequence.

    latestFrameIdx_ = curFrameIdx;
}
