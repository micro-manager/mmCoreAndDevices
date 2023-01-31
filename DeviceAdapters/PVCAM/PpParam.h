#ifndef _PPPARAM_H_
#define _PPPARAM_H_

#include <string>

/**
* Class used by post processing, a list of these elements is built up one for each post processing function
* so the call back function in CPropertyActionEx can get to information about that particular feature in
* the call back function
*/ 
class PpParam
{
public:
    PpParam(const std::string& name, short featIndex, short propIndex,
            bool isBoolean, unsigned int featId, unsigned int propId);

    const std::string& GetName() const;
    short GetFeatIndex() const;
    short GetPropIndex() const;
    bool IsBoolean() const;
    unsigned int GetFeatId() const;
    unsigned int GetPropId() const;

    unsigned int GetCurValue() const;
    void SetCurValue(unsigned int value);

protected:
    std::string mName{};
    short mFeatIndex{ -1 };
    short mPropIndex{ -1 };
    bool mIsBoolean{ false };
    unsigned int mFeatId{ (unsigned int)-1 };
    unsigned int mPropId{ (unsigned int)-1 };
    unsigned int mCurValue{ 0 };
};

#endif
