#include "PpParam.h"

PpParam::PpParam(const std::string& name, short featIndex, short propIndex,
        bool isBoolean, unsigned int featId, unsigned int propId)
    : mName(name),
    mFeatIndex(featIndex),
    mPropIndex(propIndex),
    mIsBoolean(isBoolean),
    mFeatId(featId),
    mPropId(propId)
{
}

const std::string& PpParam::GetName() const
{
    return mName;
}

short PpParam::GetFeatIndex() const
{
    return mFeatIndex;
}

short PpParam::GetPropIndex() const
{
    return mPropIndex;
}

bool PpParam::IsBoolean() const
{
    return mIsBoolean;
}

unsigned int PpParam::GetFeatId() const
{
    return mFeatId;
}

unsigned int PpParam::GetPropId() const
{
    return mPropId;
}

unsigned int PpParam::GetCurValue() const
{
    return mCurValue;
}

void PpParam::SetCurValue(unsigned int value)
{
    mCurValue  = value;
}
