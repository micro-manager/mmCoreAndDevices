#include "PropertyItem.h"

PropertyItem::PropertyItem(const std::string& name, double min, double max,
                           double step)
    : m_name(name), m_min(min), m_max(max), m_step(step) {}