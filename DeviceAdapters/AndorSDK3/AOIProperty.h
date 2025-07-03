#ifndef _AOIPROPERTY_H_
#define _AOIPROPERTY_H_

#include <map>
#include <vector>
#include "MMDeviceConstants.h"
#include "Property.h"
#include "atcore.h"
#include "IProperty.h"

class ICallBackManager;

namespace andor
{
   class ISubject;
   class IInteger;
}

class TAOIProperty : public IProperty
{
public:
   TAOIProperty(const std::string & MM_name, ICallBackManager* callback, bool readOnly);
   ~TAOIProperty();

   void Update(andor::ISubject* Subject);
   int OnAOI(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TAOIProperty> CPropertyAction;

   AT_64 GetWidth();
   AT_64 GetHeight();
   AT_64 GetLeftOffset();
   AT_64 GetTopOffset();
   unsigned GetBytesPerPixel();
   AT_64 GetStride();
   double GetBytesPerPixelF();
   void SetReadOnly(bool set_to);
   const char* SetCustomAOISize(unsigned left, unsigned top, unsigned width, unsigned height);
   const char* ResetToFullImage();

private:
   typedef std::map<long long, int> TMapAOIIndexType;
   typedef std::map<long long, long long> TMapAOIWidthHeightType;
   typedef std::map<std::string, int> TMapAOIGUIListType;
   typedef std::vector<long long> TVectorXYType;

   void populateWidthMaps(bool fullAoiControl);
   void populateLeftTopVectors();
   void populateAOIGUIList(TMapAOIIndexType::iterator iterIndex, TMapAOIWidthHeightType::iterator iter, AT_64 i64_sensorWidth);
   void findBestR2AOICoords(TMapAOIIndexType::iterator iter, AT_64 i64_sensorWidth, AT_64 i64_sensorHeight);
   void setFeature(long data);

   ICallBackManager* callback_;
   andor::IInteger* aoi_height_;
   andor::IInteger* aoi_width_;
   andor::IInteger* aoi_top_;
   andor::IInteger* aoi_left_;
   andor::IInteger* aoi_stride_;
   andor::IInteger* sensor_width_;
   andor::IInteger* sensor_height_;
   MM::Property* pbProp_;
   std::string customStr_;

   TMapAOIIndexType aoiWidthIndexMap_;
   TMapAOIWidthHeightType aoiWidthHeightMap_;
   TMapAOIGUIListType aoiGUIList_;
   TVectorXYType leftX_;
   TVectorXYType topY_;
   bool fullAoiControl_;
};

#endif // _AOIPROPERTY_H_
