#include "pch.h"
#include "actions.h"
#include "PyDevice.h"

void PyObjectAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    if (value.HasAttribute("_MM_id")) {
        auto id = value.Get("_MM_id").as<string>();
        pProp->Set(id.c_str());
    }
    else
        pProp->Set("{unknown object}");
}

PyObj PyObjectAction::get(MM::PropertyBase* pProp) {
    string id;
    pProp->Get(id);
    if (id.empty())
        return PyObj(Py_None);
    else
        return CPyHub::GetDevice(id);
}

/**
* Callback that is called when a property value is read or written
* @return MM result code
*/
int PyAction::Execute(MM::PropertyBase* pProp, MM::ActionType eAct) {
    PyLock lock;
    if (eAct != MM::BeforeGet && eAct != MM::AfterSet)
        return DEVICE_OK; // nothing to do.

    if (eAct == MM::BeforeGet) {
        // reading a property from Python
        auto value = object_.Get(attribute_.c_str());
        set(pProp, value);
    }
    else {
        object_.Set(attribute_.c_str(), get(pProp));
    }
    return DEVICE_OK;// CheckError();
}

PyBoolAction::PyBoolAction(const PyObj& object, const string& attribute, const string& MM_property) : PyAction(object, attribute, MM_property, MM::Integer) {
    enum_keys.push_back("0");
    enum_values.push_back(PyObj::Borrow(Py_False));
    enum_keys.push_back("1");
    enum_values.push_back(PyObj::Borrow(Py_True));
}

void PyBoolAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    pProp->Set(value.as<long>());
}

PyObj PyBoolAction::get(MM::PropertyBase* pProp) {
    long value;
    pProp->Get(value);
    return PyObj(value);
}

void PyFloatAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    pProp->Set(value.as<double>());
}

PyObj PyFloatAction::get(MM::PropertyBase* pProp) {
    double value;
    pProp->Get(value);
    return PyObj(value);
}

void PyIntAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    pProp->Set(value.as<long>());
}

PyObj PyIntAction::get(MM::PropertyBase* pProp) {
    long value;
    pProp->Get(value);
    return PyObj(value);
}

void PyStringAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    pProp->Set(value.as<string>().c_str());
}

PyObj PyStringAction::get(MM::PropertyBase* pProp) {
    double value;
    pProp->Get(value);
    return PyObj(value);
}

void PyEnumAction::set(MM::PropertyBase* pProp, const PyObj& value) {
    for (int i = 0; i < enum_values.size(); i++) {
        if (PyObject_RichCompareBool(enum_values[i], value, Py_EQ)) {
            pProp->Set(enum_keys[i].c_str());
            return;
        }
    }
    // value not found, do nothing
}

PyObj PyEnumAction::get(MM::PropertyBase* pProp) {
    string value;
    pProp->Get(value);
    for (int i = 0; i < enum_keys.size(); i++) {
        if (enum_keys[i] == value)
            return enum_values[i];
    }
    return PyObj(); // value not found
}
