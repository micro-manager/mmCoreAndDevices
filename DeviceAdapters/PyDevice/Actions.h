#pragma once
#include "pch.h"
#include "PyObj.h"
class CPyDeviceBase;
class PyAction : public MM::ActionFunctor {
    CPyDeviceBase* const device_;
    string attribute_; // Name of Python attribute
public:
    const string name; // Name of MM property
    const MM::PropertyType type;
    const bool readonly;
    double min = 0;
    double max = 0;
    bool has_limits = false;
    vector<string> enum_keys;
    vector<PyObj> enum_values;
public:
    PyAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, MM::PropertyType type, bool readonly) : device_(device), attribute_(attribute), name(MM_property), type(type), readonly(readonly) {}
    virtual int Execute(MM::PropertyBase* pProp, MM::ActionType eAct);
    virtual void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept = 0;
    virtual PyObj get(MM::PropertyBase* pProp)  const noexcept = 0;
};

class PyIntAction : public PyAction {
public:
    PyIntAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly) : PyAction(device, attribute, MM_property, MM::Integer, readonly) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyBoolAction : public PyAction {
public:
    PyBoolAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly);
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyFloatAction : public PyAction {
public:
    PyFloatAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly) : PyAction(device, attribute, MM_property, MM::Float, readonly) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyQuantityAction : public PyAction {
public:
    PyQuantityAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly, const PyObj& pre_set, const PyObj& post_get) : PyAction(device, attribute, MM_property, MM::Float, readonly), pre_set_(pre_set), post_get_(post_get) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
private:
    PyObj pre_set_;  // function to add ms or um unit
    PyObj post_get_; // function to remove ms or um unit
};


class PyStringAction : public PyAction {
public:
    PyStringAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly) : PyAction(device, attribute, MM_property, MM::String, readonly) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};


class PyEnumAction : public PyAction {
public:
    PyEnumAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly) : PyAction(device, attribute, MM_property, MM::String, readonly) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;

};


class PyObjectAction : public PyAction {
public:
    PyObjectAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, bool readonly) : PyAction(device, attribute, MM_property, MM::String, readonly) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

