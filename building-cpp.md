# Building On Linux

## 1. Install dependencies:

- bzip2
- bc
- build-essential
- cmake
- curl
- g++
- gfortran
- libboost-dev
- libboost-thread-dev
- libtool
- autoconf
- automake
- pkg-config
- software-properties-common
- unzip
- wget

On Debian based OSes you can install these with `apt`
```bash
sudo apt update && sudo apt install -y \
        bzip2 \
        bc \
        build-essential \
        cmake \
        curl \
        g++ \
        gfortran \
        libboost-dev \
        libboost-thread-dev \
        libtool \
        autoconf \
        automake \
        pkg-config \
        software-properties-common \
        unzip \
        wget \
```

## 2. Build and Install

```bash
./autogen.sh
./configure
make
sudo make install
```

The `sudo` is only necessary if you are installing to the default `/usr/local/micro-manager`

## 3. Download MMConfig_Demo.cfg
Many downstream libraries will expect the demo config file to be located along with the installed files. So here we download it directly from the micro-manager repo:

```bash
sudo curl https://raw.githubusercontent.com/micro-manager/micro-manager/master/bindist/any-platform/MMConfig_demo.cfg --output /usr/local/micro-manager/MMConfig_demo.cfg
```

## 4. Test install
Now downstream libraries e.g. [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus) should work.


```bash
pip install pymmcore-plus
```

```python
from pymmcore_plus import CMMCorePlus

mmcore = CMMCorePlus()
mmcore.loadSystemConfiguration("demo")
print(mmcore.getLoadedDevices())
```