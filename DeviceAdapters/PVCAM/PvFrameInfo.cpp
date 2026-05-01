#include "PvFrameInfo.h"

PvFrameInfo::PvFrameInfo()
{
}

PvFrameInfo::~PvFrameInfo()
{
}

void PvFrameInfo::SetPvHCam(short int hCam)
{
    pvHCam_ = hCam;
}

short int PvFrameInfo::PvHCam() const
{
    return pvHCam_;
}

void PvFrameInfo::SetPvFrameNr(int pvFrameNr)
{
    pvFrameNr_ = pvFrameNr;
}

int PvFrameInfo::PvFrameNr() const
{
    return pvFrameNr_;
}

void PvFrameInfo::SetPvTimeStamp(long long pvTimeStamp)
{
    pvTimeStamp_ = pvTimeStamp;
}

long long PvFrameInfo::PvTimeStamp() const
{
    return pvTimeStamp_;
}

void PvFrameInfo::SetPvReadoutTime(int pvReadoutTime)
{
    pvReadoutTime_ = pvReadoutTime;
}

int PvFrameInfo::PvReadoutTime() const
{
    return pvReadoutTime_;
}

void PvFrameInfo::SetPvTimeStampBOF(long long pvTimeStampBOF)
{
    pvTimeStampBOF_ = pvTimeStampBOF;
}

long long PvFrameInfo::PvTimeStampBOF() const
{
    return pvTimeStampBOF_;
}

void PvFrameInfo::SetTimestampMsec(double msec)
{
    timestampMsec_ = msec;
}

double PvFrameInfo::TimeStampMsec() const
{
    return timestampMsec_;
}

void PvFrameInfo::SetRecovered(bool recovered)
{
    isRecovered_ = recovered;
}

bool PvFrameInfo::IsRecovered() const
{
    return isRecovered_;
}
