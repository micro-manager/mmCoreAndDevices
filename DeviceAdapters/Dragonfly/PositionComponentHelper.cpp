#include "PositionComponentHelper.h"
#include "ASDConfigInterface.h"

using namespace std;

string CPositionComponentHelper::UndefinedPositionBase_ = "Undefined Position";

bool CPositionComponentHelper::RetrievePositionsFromFilterSet( IFilterSet* FilterSet, TPositionNameMap& PositionNames )
{
  const static unsigned int vStringLength = 64;
  bool vPositionsRetrieved = false;
  if ( FilterSet != nullptr )
  {
    unsigned int vMinPos, vMaxPos;
    if ( FilterSet->GetLimits( vMinPos, vMaxPos ) )
    {
      char vDescription[vStringLength];
      unsigned int vUndefinedIndex = 1;
      for ( unsigned int vIndex = vMinPos; vIndex <= vMaxPos; vIndex++ )
      {
        string vPositionName;
        if ( FilterSet->GetFilterDescription( vIndex, vDescription, vStringLength ) == false )
        {
          vPositionName += UndefinedPositionBase_ + " " + to_string(vUndefinedIndex);
          vUndefinedIndex++;
        }
        else
        {
          vPositionName += vDescription;
        }
        PositionNames[vIndex] = vPositionName;
      }

      vPositionsRetrieved = true;
    }
  }
  return vPositionsRetrieved;
}

void CPositionComponentHelper::RetrievePositionsWithoutDescriptions( unsigned int MinValue, unsigned int MaxValue, TPositionNameMap& PositionNames )
{
  for ( unsigned int vIndex = MinValue; vIndex <= MaxValue; vIndex++ )
  {
    string vPositionName = UndefinedPositionBase_ + " " + to_string( vIndex );
    PositionNames[vIndex] = vPositionName;
  }
}
