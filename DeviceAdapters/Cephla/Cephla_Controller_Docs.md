
# The Cephla Squid Controller
The Squid Micro-controller by the company Cephla controls multiple microscope components such as motorized XY stages, motorized Z stages, LED and laser light source, and it provides analog (0-5V) and digital outputs.  Internally, the controller consists of an Arduino (Due in earlier versions, Teensy 4.1 in newer versions) board that communicates to several other boards, such as motor controllers ([TMC4361](https://www.analog.com/en/products/tmc4361.html), and [TMC2660](https://www.analog.com/en/products/TMC2660.html#part-details)), and DAC boards ([DAC 80508](https://www.ti.com/product/DAC80508) from TI). The design and firmware for the controller are [open source,](https://github.com/hongquanli/octopi-research) hence can be modified for everyone's needs. 

## Cabling
The unit needs a ?V power supply with ?A minimum.  Most connections are straight forward (XY stage cable, Z Stage cable).  The USB connection to the computer is unconventional, i.e. you need a USB cable with ? on both ends.  Communication is through a (virtual) serial port, and the device will show up as a COM port in the Windows Device Manager.  Baud rate is ignored and communication will take place at highest speeds allowed by the USB connection (i.e. you can choose whatever number for the baud rate). 

## Communication protocol



