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
    const bool readOnly_ = false;
    double min = 0;
    double max = 0;
    bool has_limits = false;
    vector<string> enum_keys;
    vector<PyObj> enum_values;
public:
    PyAction(CPyDeviceBase* device, const string& attribute, const string& MM_property, MM::PropertyType type) : device_(device), attribute_(attribute), name(MM_property), type(type) {}
    virtual int Execute(MM::PropertyBase* pProp, MM::ActionType eAct);
    virtual void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept = 0;
    virtual PyObj get(MM::PropertyBase* pProp)  const noexcept = 0;
};

class PyIntAction : public PyAction {
public:
    PyIntAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::Integer) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyBoolAction : public PyAction {
public:
    PyBoolAction(CPyDeviceBase* device, const string& attribute, const string& MM_property);
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyFloatAction : public PyAction {
public:
    PyFloatAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::Float) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyStringAction : public PyAction {
public:
    PyStringAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};


class PyEnumAction : public PyAction {
public:
    PyEnumAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;

};


class PyObjectAction : public PyAction {
public:
    PyObjectAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

