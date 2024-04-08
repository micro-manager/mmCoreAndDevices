#include "pch.h"
#include "Actions.h"
#include "PyDevice.h"

/**
* Callback that is called when a property value is read or written
* @return MM result code
*/
int PyAction::Execute(MM::PropertyBase* pProp, MM::ActionType eAct) {
    PyLock lock;
    if (eAct != MM::BeforeGet && eAct != MM::AfterSet)
        return DEVICE_OK; // nothing to do.

    if (eAct == MM::BeforeGet) {
        auto value = getter_.Call();
        set(pProp, value);
    }
    else 
        setter_.Call(get(pProp));
    
    return check_errors_();
}

void PyBoolAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    pProp->Set(value.as<long>());
}

PyObj PyBoolAction::get(MM::PropertyBase* pProp) const noexcept {
    long value;
    pProp->Get(value);
    return PyObj(value);
}

void PyFloatAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    pProp->Set(value.as<double>());
}

PyObj PyFloatAction::get(MM::PropertyBase* pProp) const noexcept {
    double value;
    pProp->Get(value);
    return PyObj(value);
}

void PyIntAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    pProp->Set(value.as<long>());
}

PyObj PyIntAction::get(MM::PropertyBase* pProp) const noexcept {
    long value;
    pProp->Get(value);
    return PyObj(value);
}

void PyStringAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    pProp->Set(value.as<string>().c_str());
}

PyObj PyStringAction::get(MM::PropertyBase* pProp) const noexcept {
    string value;
    pProp->Get(value);
    return PyObj(value);
}

void PyEnumAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    for (int i = 0; i < enum_values.size(); i++) {
        if (PyObject_RichCompareBool(enum_values[i], value, Py_EQ)) {
            pProp->Set(enum_keys[i].c_str());
            return;
        }
    }
    // value not found, do nothing
}

PyObj PyEnumAction::get(MM::PropertyBase* pProp) const noexcept {
    string value;
    pProp->Get(value);
    for (int i = 0; i < enum_keys.size(); i++) {
        if (enum_keys[i] == value)
            return enum_values[i];
    }
    return PyObj(); // value not found
}
