## Check bootstrap.py to see the requirements in code. 
Note that you can always expose more than only the required properties to MicroManager, which will show in the device manager.
For these required properties and methods, you need to use the EXACT naming structure, such that PyDevice can translate them into MicroManager devices.

## Available devices types:
1. Device
2. Camera
3. Stage
4. XYStage
5. SLM

## 1. Device
This is the default device type, if none of the requirements for a specific device are met, the device type will be set to this.

## 2. Camera
Required properties:
    data_shape: tuple[int]
    measurement_time: Quantity[u.ms]
    top: int
    left: int
    height: int
    width: int
Required methods:
    trigger(self) -> None
    read(self) -> np.ndarray

Simple example is shown in test.py 

## 3. Stage
Required properties:
    step_size: Quantity[u.um]
    position: Quantity[u.um]
Required methods:
    home(self) -> None
    wait(self) -> None

## 4. XYStage
Required properties:
    x: Quantity[u.um]
    y: Quantity[u.um]
    step_size_x: Quantity[u.um]
    step_size_y: Quantity[u.um]
Required methods:
    home(self) -> None
    wait(self) -> None

## 5. SLM
Required properties:
    phases: np.ndarray
Required methods:
    update(self, wait_factor=1.0, wait=True)
    wait(self) -> None
    reserve(self, time: Quantity[u.ms])