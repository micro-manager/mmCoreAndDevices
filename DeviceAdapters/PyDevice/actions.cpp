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
        auto value = device_->Object().Get(attribute_.c_str());
        set(pProp, value);
    }
    else 
        device_->Object().Set(attribute_.c_str(), get(pProp));
    
    return device_->CheckError();
}


void PyObjectAction::set(MM::PropertyBase* pProp, const PyObj& value) const noexcept {
    if (value.HasAttribute("_MM_id")) {
        auto id = value.Get("_MM_id").as<string>();
        pProp->Set(id.c_str());
    }
    else
        pProp->Set("{unknown object}");
}

PyObj PyObjectAction::get(MM::PropertyBase* pProp) const noexcept {
    string id;
    pProp->Get(id);
    if (id.empty())
        return PyObj(Py_None);
    else
        return CPyHub::GetDevice(id);
}


PyBoolAction::PyBoolAction(CPyDeviceBase* device, const string& attribute, const string& MM_property) : PyAction(device, attribute, MM_property, MM::Integer) {
    enum_keys.push_back("0");
    enum_values.push_back(PyObj::Borrow(Py_False));
    enum_keys.push_back("1");
    enum_values.push_back(PyObj::Borrow(Py_True));
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
    double value;
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
