#pragma once

#include "PVCAMIncludes.h"

/**
* A local definition for the FRAME_INFO. Used to copy the
* essential FRAME_INFO metadata together with additional frame metadata.
*/
class PvFrameInfo
{
public:
    PvFrameInfo();
    ~PvFrameInfo();

    // FRAME_INFO related metadata

    void SetPvHCam(short int hCam);
    short int PvHCam() const;

    void SetPvFrameNr(int pvFrameNr);
    int PvFrameNr() const;

    void SetPvTimeStamp(long long pvTimeStamp);
    long long PvTimeStamp() const;

    void SetPvReadoutTime(int pvReadoutTime);
    int PvReadoutTime() const;

    void SetPvTimeStampBOF(long long pvTimeStampBOF);
    long long PvTimeStampBOF() const;

    // Other metadata added by the adapter

    void SetTimestampMsec(double msec);
    double TimeStampMsec() const;

    void SetRecovered(bool recovered);
    bool IsRecovered() const;

private:
    // FRAME_INFO Metadata
    short int pvHCam_{ 0 };         // int16  FRAME_INFO.hCam
    int       pvFrameNr_{ 0 };      // int32  FRAME_INFO.FrameNr
    long long pvTimeStamp_{ 0 };    // long64 FRAME_INFO.TImeStamp (EOF)
    int       pvReadoutTime_{ 0 };  // int32  FRAME_INFO.ReadoutTime
    long long pvTimeStampBOF_{ 0 }; // long64 FRAME_INFO.TimeStampBOF

    // Additional Metadata
    double    timestampMsec_{ 0 };   // MM Timestamp
    bool      isRecovered_{ false }; // Recovered from missed callback
};
