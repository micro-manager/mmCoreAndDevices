#pragma once
#include "pch.h"
#include "PyObj.h"
class PyAction : public MM::ActionFunctor {
    PyObj object_; // Python object
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
    PyAction(const PyObj& object, const string& attribute, const string& MM_property, MM::PropertyType type) : object_(object), attribute_(attribute), name(MM_property), type(type) {}
    virtual int Execute(MM::PropertyBase* pProp, MM::ActionType eAct);
    virtual void set(MM::PropertyBase* pProp, const PyObj& value) = 0;
    virtual PyObj get(MM::PropertyBase* pProp) = 0;
};

class PyIntAction : public PyAction {
public:
    PyIntAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::Integer) {}
    void set(MM::PropertyBase* pProp, const PyObj& value);
    PyObj get(MM::PropertyBase* pProp);
};

class PyBoolAction : public PyAction {
public:
    PyBoolAction(const PyObj& object, const string& attribute, const string& MM_property);
    
    virtual void set(MM::PropertyBase* pProp, const PyObj& value);
    virtual PyObj get(MM::PropertyBase* pProp);
};

class PyFloatAction : public PyAction {
public:
    PyFloatAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::Float) {}
    void set(MM::PropertyBase* pProp, const PyObj& value);
    PyObj get(MM::PropertyBase* pProp);
};

class PyStringAction : public PyAction {
public:
    PyStringAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value);
    PyObj get(MM::PropertyBase* pProp);
};


class PyEnumAction : public PyAction {
public:
    PyEnumAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value);
    PyObj get(MM::PropertyBase* pProp);

};


class PyObjectAction : public PyAction {
public:
    PyObjectAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::String) {}
    void set(MM::PropertyBase* pProp, const PyObj& value);
    PyObj get(MM::PropertyBase* pProp);
};

