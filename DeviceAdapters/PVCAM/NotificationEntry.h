#pragma once

#include "PvFrameInfo.h"

/**
* A class that contains a pointer to frame data and corresponding
* frame metadata. This class is used by the NotificationThread.
*/
class NotificationEntry
{
public:

    NotificationEntry();
    explicit NotificationEntry(
            const void* pData, unsigned int dataSz, const PvFrameInfo& metadata);

    /**
    * Returns the frame metadata
    * @return Frame metadata
    */
    const PvFrameInfo& FrameMetadata() const;
    /**
    * Return the pointer to the frame data.
    * @return address of the frame data
    */
    const void* FrameData() const;

    /**
    * Returns the size of the frame data in bytes
    * @return Frame data size in bytes
    */
    unsigned int FrameDataSize() const;

private:
    const void*  pFrameData_{ nullptr }; ///< Pointer to the frame in circular buffer
    unsigned int frameDataSz_{ 0 }; ///< Size of the data in bytes
    PvFrameInfo frameMetaData_{}; ///< Copy of the frame metadata
};
