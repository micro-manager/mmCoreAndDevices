# Conda environment

conda create --name mmcore

# mmCoreAndDevices
The c++ code at the core of the Micro-Manager project.

## API Docs
[Main Page](https://micro-manager.org/apidoc/MMCore/latest/index.html)

If you are using a scripting language to control a microscope through the CMMCore object
then you are likely looking for the [CMMCore API](https://micro-manager.org/apidoc/MMCore/latest/class_c_m_m_core.html)

## Build

I recommend building inside a conda environment:

```
conda create --name mmcore
conda activate mmcore   
```

Then:
```
cmake .
make
make install
```