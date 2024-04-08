#pragma once
#include "pch.h"
#include "PyObj.h"

class PyAction : public MM::ActionFunctor {
    PyObj getter_;
    PyObj setter_;
    ErrorCallback check_errors_;
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
    PyAction(const PyObj& getter, const PyObj& setter, const string& name, MM::PropertyType type, const ErrorCallback& callback) : getter_(getter), setter_(setter), name(name), type(type), readonly(!setter_), check_errors_(callback) {}
    virtual int Execute(MM::PropertyBase* pProp, MM::ActionType eAct);
    virtual void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept = 0;
    virtual PyObj get(MM::PropertyBase* pProp)  const noexcept = 0;
};

class PyIntAction : public PyAction {
public:
    PyIntAction(const PyObj& getter, const PyObj& setter, const string& name, const ErrorCallback& callback) : PyAction(getter, setter, name, MM::Integer, callback) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyBoolAction : public PyAction {
public:
    PyBoolAction(const PyObj& getter, const PyObj& setter, const string& name, const ErrorCallback& callback) : PyAction(getter, setter, name, MM::Integer, callback) {
        enum_keys.push_back("0");
        enum_values.push_back(PyObj(false));
        enum_keys.push_back("1");
        enum_values.push_back(PyObj(true));
    }
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyFloatAction : public PyAction {
public:
    PyFloatAction(const PyObj& getter, const PyObj& setter, const string& name, const ErrorCallback& callback) : PyAction(getter, setter, name, MM::Float, callback) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyStringAction : public PyAction {
public:
    PyStringAction(const PyObj& getter, const PyObj& setter, const string& name, const ErrorCallback& callback) : PyAction(getter, setter, name, MM::String, callback) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

class PyEnumAction : public PyAction {
public:
    PyEnumAction(const PyObj& getter, const PyObj& setter, const string& name, const ErrorCallback& callback) : PyAction(getter, setter, name, MM::String, callback) {}
    void set(MM::PropertyBase* pProp, const PyObj& value) const noexcept override;
    PyObj get(MM::PropertyBase* pProp) const noexcept override;
};

