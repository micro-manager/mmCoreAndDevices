/*
This Arduino Sketch is based on the one written by   
//  Bonno Meddens, 30/07/2019
For 32-Bit microcontroller boards.

This Version Written by:
Dr Justin E. Molloy
Professor of Biophysics
Centre for Mechanochemical Cell Biology (L1.01)
Warwick Medical School
Coventry CV4 7AL
M: +44 (0) 7986143550
E: justin.molloy@warwick.ac.uk

The code interfaces with MicroManager Adapter called mmgr_dal_ESP32.dll
The C++ code for the MM adapter is also deposited on GitHUb.
ESP32.Cpp & ESP32.h

This arduino sketch is highly modified and specifically for ESP32 series microcontrollers.

If you want to use Bluetooth, Wifi, Serial etc.. you will need to allocate more memory on your ESP Chip.

To compile this program for NodeMCU32s using Arduino IDE modify the memory allocation file:
c:\Users\username\AppData\Local\Arduino15\packages\esp32\hardware\esp32\2.0.9\tools\partitions\default.csv
You must increase App0 to at least 0x190000 
(Reduce App1 by same amnount and change memory offset start addresses)
Note: App1 memory is usually reserved for "Over the Air" updates... we don't need that!
As below:
# Name    Type   SubType     Offset       Size     Flags
nvs       data     nvs       0x9000    0x5000	
otadata   data     ota       0xe000    0x2000	
app0       app    ota_0     0x10000  0x190000	
app1       app    ota_1    0x1A0000  0x0F0000	
spiffs    data    spiffs   0x290000  0x160000	
coredump  data   coredump  0x3F0000   0x10000	
====================================================================

And .....
c:\Users\username\AppData\Local\Arduino15\packages\hardware\esp\1.0.6\boards.txt
search for....nodemcu and change "maximum_size=1638400" (=0x190000)
nodemcu-32s.upload.tool=esptool_py
nodemcu-32s.upload.maximum_size=1638400
nodemcu-32s.upload.maximum_data_size=327680
nodemcu-32s.upload.wait_for_upload_port=true

// Notes on the Bonno Meddens' original Arduino  code:
//////////////////////////////////////////////////////
The old MM-Device Adaptor code: "ARDUINO32bitBoards.cpp"
Used the following commands:
Command Number	=	Associated Method/Class
	1 	=   CArduino32Shutter::WriteToPort(long value)
	1 	= 	CArduino32Switch::WriteToPort (long value)
	3 	= 	CArduino32DA::WriteToPort(unsigned long value)
	5&6	=   CArduino32Switch::LoadSequence(unsigned size, unsigned char* seq)
	8&9	=   CArduino32Switch::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
	11 	= 	CArduino32Switch::OnRepeatTimedPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
	12&9 =  CArduino32Switch::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
	20&21= 	CArduino32Switch::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
	22	= 	CArduino32Switch::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
	30&31	= CArduino32Hub::GetControllerVersion(int& version) 
	40	= 	CArduino32Input::GetDigitalInput(long* state)
	4	  =   CArduino32Input::OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType eAct, long  channel )***check***
	42	= 	CArduino32Input::SetPullUp(int pin, int state)

These old codes seemed to be unused:
2 - Query digitial outputs 
7 - Skip triggers
10 - Time intervals for trigger mode
13-19 - Not defined
23-29, 32-39, 41, 43->

All the old commands are now replaced with ASCII command and value format:
=================================================================
  Comma-delimited commands and parameters
  Example command format:	"P,7,123"
	gives:	triggerPattern[7]= 0111 1011 Binary
================================================================= 
ESP32 Board pin-out....
==========================================================================================
                                    NodeMCU ESP32S
                              3.3V o-           -o GND
                             Reset o-           -o GPIO23      * SDA - to OLED display
                      ADC-5 GPIO36 o-           -o GPIO22      * SCL - to OLED display
                      ADC-4 GPIO39 o-           -o Int
                      ADC-3 GPIO34 o-           -o Int
                      ADC-2 GPIO35 o-           -o GPIO21     
                      ADC-1 GPIO32 o-           -o GND
                      ADC-0 GPIO33 o-           -o GPIO19      *Shutter - to FET driver 1A max. SHUTTER
    X OpAmp+LP filter *ch5  GPIO25 o-           -o GPIO18      *ch3 - to FET LED driver 1A max. Blue
    Y OpAmp+LP filter *ch6  GPIO26 o-           -o GPIO5       *ch2 - to FET LED driver 1A max. Green
    Z OpAmp+LP filter *ch7  GPIO27 o-           -o GPIO17      *ch1 - to FET LED driver 1A max. Yellow
                            GPIO14 o-           -o GPIO16      *ch0 - to FET LED driver 1A max. Red
                            GPIO12 o-           -o GPIO4        
                               GND o-           -o GPIO0       {O/P only - affects boot}
                            GPIO13 o-           -o GPIO2
                            Int 09 o-           -o GPIO15      *inPin <= blanking deglitch with 0.1uF cap. to GND  
                            Int 10 o-           -o Int GPIO8
                            Int 11 o-           -o Int GPIO7
                                5V o-   |   |   -o Int GPIO6
                                        |___|
                                       USB skt 
==========================================================================================
Note 8-bit res on GPIO25/26 is pretty poor so maybe set them to PWM mode at 14 bit res, (clk ~10kHz)
Need to low-pass filter to about 10Hz to remove ripple - which would be fine for most X-Y stages
Better still, use GPIO 25,26,27 as chip select lines for SPI comms to 16 or 20bit DACS.
*/

// BLUETOOTH
// Mobile APP for communicating with microcontrollers 
// Available at: WWW.KEUWL.COM
// You can set up sliders, buttons, pad, dials etc
#include "BluetoothSerial.h"
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif
BluetoothSerial SerialBT;

// WiFi Stuff
#include <WiFi.h>
// create a server and listen on port 8088
WiFiServer server(8088);

//local login credentials at work
//const char* ssid = "WorkRouter";
//const char* password = "AnotherPASSWORD";

// local login credentials at home
const char* ssid = "HOMERouter";
const char* password = "PASSWORD";

// Stuff for the I2C OLED display
//#include <Wire.h>
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128     // OLED display width, in pixels
#define SCREEN_HEIGHT 64     // OLED display height, in pixels
#define OLED_RESET -1        // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C  ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
// define GPIO pins for I2C communications
#define I2C_SDA 23
#define I2C_SCL 22
int linemax = 8;
#define dummy "                     "
// initialise nlines string values..for pointer to index!
String txtlines[] = { dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy };

String MMresponse = "MM-ESP32";   // Expected response from Arduino
unsigned int version = 1;         // This is Version 1 <= update as necessary to allow checking by MM

const int SEQUENCELENGTH = 12;    // This should be good enough

// Each byte in triggerPattern[] array codes the output blanking pattern for upto 5 - channels
// Each entry is bitmapped as a binary mask to turn output pins off (="0") or at Vout value (="1").
// bits 7,6,5 are reserved... bit4=shutter, bit3=blue, bit2=green, bit1=yellow, bit0=red.
// Initially all outputs are active for every timed sequence position e.g. all "r r r 1 1 1 1 1 BIN" = 255 DEC
byte       triggerPattern[SEQUENCELENGTH] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 };

//Delay times in milliseconds for each sequence output
unsigned int triggerDelay[SEQUENCELENGTH] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

//We can also "sequence" the x,y,z stage positions
//Here, we store the x,y,z positions 
unsigned int   xPosn[SEQUENCELENGTH] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
unsigned int   yPosn[SEQUENCELENGTH] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
unsigned int   zPosn[SEQUENCELENGTH] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

int patternLength = 0;      // Number of pattern slots that are currently in use
byte repeatPattern = 0;     // Number of repeat cycles

volatile long triggerNr;    // # of triggers can start at negative value = "-skipTriggers"
volatile long sequenceNr;   // current sequence number as we cycle through sequence pattern (0->patternLength)

int skipTriggers = 0;       // # of triggers to skip before starting to generate patterns (triggerNr is set to "-skipTriggers")
byte currentPattern = 255;  // A global variable that keeps track of channel output logic

bool blanking = false;
bool blankOnHigh = false;
bool triggerMode = false;
bool triggerState = false;

// Some thoughts on using PWM for Stage control and LED control: 
// NodeMCU 32S PWM clock (APB clk) runs at 80MHz (regardless of CPU)
// Max possible PWM freq = 80,000,000/(2^nbits)
//
//  "bit noise"      resolution    Max. PWM freq      
// Amax*0.015e-3       16 bit     ~  1.2  kHz  (we use 1kHz)
// Amax*0.03e-3        15 bit     ~  2.4  kHz  (we use 2kHz)
// Amax*0.06e-3        14 bit     ~  4.88 kHz  (we use 4kHz)
// Amax*0.24e-3        12 bit     ~ 19.5  kHz  (we use 16kHz)
// Amax*0.98e-3        10 bit     ~ 78.1  kHz  (we use 50kHz)
//
// If Amax is 200um range on a piezo stage multiply by 200,000 to get nm/bit
// Now, if you use a single-pole R-C filter on the PWM output the 
// PWM ripple equivalent will be very approx 200,000 * (fc/PWMfreq) nm "peak-peak"
// 
// e.g. at 12-bit PWM resolution:
// The smallest stepsize = 0.24e-3*200,000 = 48nm/bit
// And using a 15 kOhm + 1 uF "R-C" filter = fc~10Hz (100ms response time).
// PWM ripple noise = (10 Hz/16,000 Hz) * 200,000 = 125nm P-P YUK!!!!
// RESULT, PWM is fine for testing but use 20-bit DACs for real applications
//
// However, 10-bit PWM control for LEDs is fine because 
// if your max frame rate is 100Hz (10ms/frame) and the PWM 
// is asynchronous with the camera frames you will get 500 LED flashes +/-1
// on each frame so "flicker" (1/500) is trivial.
//
//                     chan   0   1   2   3   
const uint8_t  DACpin[]  = { 16, 17,  5, 18 }; // GPIO Pins (OUTPUT)
const uint32_t PWMfreq[] = { 50, 50, 50, 50 }; // PWM freq in kHz
const uint32_t PWMres[]  = { 10, 10, 10, 10 }; // PWM resolution in bits
unsigned int Vout[]      = {  0,  0,  0,  0 }; // LED output values
const int nDAC           = 4; // 4 DAC channels

// Stage Position control
const uint8_t  xPin     =  25; // GPIO Pin (OUTPUT)  
const uint32_t xPWMfreq =   4; // PWM freq in kHz
const uint32_t xPWMres  =  14; // PWM resolution in bits
unsigned int xOut       =   0; // xyz Stage output values
const int xChan         =  10; // PWM output channel number
int xRange              = 200; // xRange (um)

const uint8_t  yPin     =  26; // GPIO Pin (OUTPUT)  
const uint32_t yPWMfreq =   4; // PWM freq in kHz
const uint32_t yPWMres  =  14; // PWM resolution in bits
unsigned int yOut       =   0; // xyz Stage output values
const int yChan         =  11; // PWM output channel number
int yRange              = 200; // yRange (um)
   
const uint8_t  zPin     =  27; // GPIO Pin (OUTPUT)  
const uint32_t zPWMfreq =   4; // PWM freq in kHz
const uint32_t zPWMres  =  14; // PWM resolution in bits
unsigned int zOut       =   0; // xyz Stage output values
const int zChan         =  12; // PWM output channel number
int zRange              = 100; // zRange (um)

// Analogue input GPIO pins on the ESP32 NODEMCU
//                          0   1   2   3   4   5
const uint8_t ADCpin[] = { 33, 32, 35, 34, 39, 36 };  // set GPIO Pins (INPUT)
const int         nADC = 6; // 6 ADC channels

//"Logical" pin for blanking (I/P) and harware shutter (O/P)
const uint8_t shutterPin = 19; // set GPIO Pin for Shutter control (O/P)
const uint8_t inPin      = 15; // set GPIO Pin for trigger/blanking (I/P)

// Create some global variable so we can see the commands + params everywhere
// This saves passing values back-and-forth to functions all the time!
int const maxVal=4;
int const maxBytes=50;
char command = 0; 
char inBuf[maxBytes];
char outBuf[maxBytes];
float val[maxVal];
int ct=0;

//============================= Setup ===============================
void setup() {

Serial.begin(115200);  //Baud rate

  Wire.begin(I2C_SDA, I2C_SCL);
// SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);

// Clear the buffer
  display.clearDisplay();
  display.setTextSize(1);                              // Normal 1:1 pixel scale
  display.setTextColor(SSD1306_WHITE, SSD1306_BLACK);  // Draw white text, Black background

// Set inPin for input (pullup=normally high)
  pinMode(inPin, INPUT_PULLUP);
  // Set Shutter for output
  pinMode(shutterPin, OUTPUT);

// Configure the PWM ouptut channels 
  ledcSetup(xChan, xPWMfreq*1000, xPWMres);
  ledcSetup(yChan, yPWMfreq*1000, yPWMres);
  ledcSetup(zChan, zPWMfreq*1000, zPWMres);
// Assign the correct GPIO pins to the channels 

// xChan, yChan, zChan are RESERVED 10, 11, 12
  ledcAttachPin(xPin, xChan);  
  ledcAttachPin(yPin, yChan);
  ledcAttachPin(zPin, zChan);

// PWM setup - ESP32 has up to 16 independent PWM channels
// DACPWM channels 0->nDac (where nDAC<10 - see x,y,z chans above)
  for (uint8_t i = 0; i < nDAC; i++) ledcSetup(i, PWMfreq[i]*1000, PWMres[i]);
  for (uint8_t i = 0; i < nDAC; i++) ledcAttachPin(DACpin[i], i);

  String outStr = "Running:" + MMresponse + "-v" + String(version);  
  writeln(outStr);
  
 // clear the RS232 serial buffer   
  Serial.flush();

// Start Bluetooth
  SerialBT.begin("Node_ESP32s"); //Bluetooth device name
  writeln("Bluetooth=Node_ESP32s");

// connect to WiFi
  outStr = "Connect:" + String(ssid);
  writeln(outStr);
  
  WiFi.begin(ssid, password);
  delay(100);
  
  boolean connected= false;
  int ct = 0;
// wait until ESP32 connected
  while ((WiFi.status() != WL_CONNECTED) && (ct<20)) {
    delay(100);
    ct+=1;
  }

  if (ct<20) connected = true;
  
  IPAddress ip = WiFi.localIP();
  char buf[16];
  sprintf(buf, "%u.%u.%u.%u", ip[0], ip[1], ip[2], ip[3]);
  outStr = "IP:" + String(buf);
  writeln(outStr);

// start Server
  if (connected) server.begin();

}
//========================= Setup ENDS ===========================

//===========
void loop() {
//===========
// Sit in this main loop and wait for incoming commands over RS232 (USB)

 WiFiClient client = server.available();
  if (client) {
    while (client.connected()) { 

// check for WiFi input
      if (client.available() > 0) {
        delay(5);  // wait for WiFi buffer to fill
        int nbytes = client.available();
        if (nbytes > maxBytes) nbytes = maxBytes;
        int len = client.readBytesUntil('\n', inBuf, nbytes);
        inBuf[len]=0;
        String str = String(inBuf);
        splitStr(str);
        // we now have the command character and the parameters val[0]->val[3]
        bool respond = doCommand();
        // We MUST send the outBuf if respond is true
        if (respond) {
          client.print(outBuf);
        }  
        if (command != 'L') writeln(String(outBuf));                 
        command = 0;
      }

    if (Serial.available() > 0) doRS232();       // check for RS232 input

    if (SerialBT.available() > 0) doBlueTooth(); // check for bluetooth

// check for trigger or blanking during Video recording
    checkTrig();
   } // loop while WiFI client connected  
  }  // WiFi client no longer connected...

  if (Serial.available() > 0) doRS232();         // check for RS232 input

  if (SerialBT.available() > 0) doBlueTooth();   // check for bluetooth

  checkTrig();
  
// loop and try for new WiFi client
}

//========================= Main Loop ENDS =========================
//==============
void doRS232() {
//==============  
// new RS232 data available
    delay(50); 
    int nbytes = Serial.available();
    if (nbytes > maxBytes) nbytes = maxBytes;
    int len = Serial.readBytesUntil('\n', inBuf, nbytes);  
    inBuf[len]=0;
    String str = String(inBuf);
    splitStr(str);
    bool respond = doCommand();
    // We MUST send the outBuf if respond is true
    if (respond){
      Serial.print(outBuf);
    }     
    if (command != 'L') writeln(String(outBuf));        
    command=0;
}
//==================
void doBlueTooth() {
//================== 
// new Bluetooth data available
    delay(50); //wait for buffer to fill
    int nbytes = SerialBT.available();
    if (nbytes > maxBytes) nbytes = maxBytes;
    int len = SerialBT.readBytesUntil('\n', inBuf, nbytes);  
    inBuf[len]=0;
    String str = String(inBuf);
    splitStr(str);
    bool respond = doCommand();
    // We MUST send the outBuf if respond is true.
    if (respond){ 
      //Serial.print(outBuf);       
    }
    if (command != 'L') writeln(String(outBuf));
    command=0;  
}

//================
void checkTrig() { 
//================  
// On eack loop check for a trigger signal
// In trigger mode, we will blank even if blanking is not on..
// Only do this on first transition of the trigger signal
      if (triggerMode) {
        bool trigSig = digitalRead(inPin);
        if ((triggerState != trigSig) && (blankOnHigh == trigSig)) { // first trigger - turn everything off
          writeZeros();
          triggerState = trigSig; 
        } else if (blankOnHigh == trigSig) { // subsequent triggers
        if (triggerNr >= 0) {  // TRUE when we have skipped "n" pre-triggers
          writePattern(triggerPattern[sequenceNr]);                                    // move to next pattern
          sequenceNr = (sequenceNr++) % patternLength;          // modulus wrap around
        }
        triggerNr++;  // tally number of triggers
        }
      } else if (blanking) {  //   blanking mode??
        if ((blankOnHigh == digitalRead(inPin)))  writePattern(currentPattern);
        else writeZeros();
      }   
 }
 
//=========================
void splitStr(String str) {
//=========================
// This is called when a string has been received from RS232, TelNet or Bluetooth
// We leave the function having updated:
// "command" string (usually just one character) and variables "val[0], val[1], val[2]...
//  writeln("Recd:" +str);
  str.trim();
  int startIndex = 0;
  int commaIndex = str.indexOf(',', startIndex);         // Search for delimiter ","
  if (commaIndex == -1) commaIndex = str.length();       // Command with no numerical parameters
  String param = str.substring(startIndex, commaIndex);  // e.g. "m" or "MOV" (allows long command names)
  str.toUpperCase();
  command = str[0];  // e.g. "mov" => "MOV" => "M"
// work through the string and extract parameters (up to 4) "M,12.34,56.7,89.1"
  int nVals = 0;
  while ((commaIndex > 0) && (nVals < maxVal)) {  // we have at lease one comma and presumably another variable?
    startIndex = commaIndex + 1;
    commaIndex = str.indexOf(',', startIndex);  // get params
    if (commaIndex > 0) {
      param = str.substring(startIndex, commaIndex);
      val[nVals] = param.toFloat();
      startIndex = commaIndex + 1;
    } else {
      param = str.substring(startIndex, str.length());
      val[nVals] = param.toFloat();
    }
    nVals += 1;
  }
}

//================
bool doCommand() {
//================  
// This function executes the received command
// It uses global variables "command" and "val[0],val[1]..."
// A bit messy but economical if we have different control inputs:
// RS232, TelNet, BlueTooth, and possibly user buttons etc changing things!
// "X,Y,Z" commands are for stage movements
//================
bool resp=false;
unsigned int chan = 0;
unsigned int seqNo = 0;
float posn = 0;
unsigned int state = 0;
unsigned int tmp = 0;

 switch (command) {

// "V" = Return Firmware ID and version number (was 30&31)
// RETURN "MM-ESP32",version#"+crlf 
    case 'V':
      // we must always send this response - do not disable  
      sprintf(outBuf,"%s,%d\r\n",MMresponse,version); 
      resp=true;
      break;

// "H,0/1" = sHutter open/close <= this can be a mechanical device or 
// as we do here we use electronic shuttering of the LED output pins
    case 'H':  
      if (val[0] == 0){ 
        digitalWrite(shutterPin,LOW); // Close Shutter (assumes "normally closed")
        writeZeros();
      } else if (val[0] > 0) {
        currentPattern = (byte) val[0];
        writePattern(currentPattern);
        digitalWrite(shutterPin,HIGH); // OPEN Shutter (assumes "normally closed")
      }
      sprintf(outBuf,"H,%d\r\n",(byte) val[0]);
      break;

// "B,1/0" = Start Blanking mode based on InPin gating signal (was 20 & 21)
// Outputs = 0 while "InPin" LOW
// Outputs = command "S" pattern while "InPin" HIGH
    case 'B':
      if (val[0]==0) blanking = false;
      else blanking = true;
      sprintf(outBuf,"B,%d\r\n",val[0]); 
      break;

// "F,0/1" = Flip 'polarity' of input TTL for blanking mode
// Note: 1 means pin is "active High" (blankOnHigh=true) (i.e. the pin is normally low)
//       0 is "active Low" (BlankonHigh=false) (default) (i.e. pin is normally held high)
// By default we make it "active low" (so set pin Pull-up "ON") which is the norm for digital circuits.    
    case 'F':
      if (val[0] == 1) blankOnHigh = false;
      else blankOnHigh = true;
      sprintf(outBuf,"F,%d\r\n",val[0]); 
      break;

// "S" = Set digital output pattern (was 1)
// "S,v"  v is desired digital pattern
    case 'S':  
      currentPattern = (byte) val[0];
      if (!blanking) writePattern(currentPattern);
      sprintf(outBuf,"S,%d\r\n",val[0]);
      break;

// "Q" = Query digital output pattern (was 2)
// duplex - RETURN "Q,p"+crlf Where p is digital output pattern
// should always return "p"+crlf
    case 'Q':
      sprintf(outBuf,"Q,%d\r\n",currentPattern);
      resp=true;   
      break;

// "O" = Output Vout value to channel (was 3)
// "O,chan,%OP" - ouput channel and %O/P as floating point
    case 'O':
      chan = (int) val[0];
      if (val[1]>100) val[1]=100;
      if (val[1]<0) val[1]=0;
      if (chan < nDAC) {        
        Vout[chan] = val[1]/100 * (1 << PWMres[chan]);
        ledcWrite(chan, Vout[chan]);
      }
      sprintf(outBuf,"O,%d,%3.2f\r\n",val[0],val[1]);
      break;

// "P" = bit Pattern used in triggered mode (was 5)
// "P,n,d"  - n is pattern index (0->12)
//          - d is digital pattern
    case 'P':    
      seqNo = (int) val[0];
      if ((seqNo >= 0) && (seqNo < SEQUENCELENGTH)) {        
        triggerPattern[seqNo] = (byte) val[1];
        char byte2bin[8];
// print bit pattern to the OLED screen  
        for (int i = 0; i < 8; i++) {
          byte2bin[i] = ((byte) triggerPattern[seqNo] &  ((byte) 1 << (7-i))) ? '1' : '0';            
        }
          char pBuf[22];
          sprintf(pBuf, "O/P#:%d = %s", seqNo ,String(byte2bin));
          writeln(String(pBuf));
      }
      sprintf(outBuf,"P,%d,%d\r\n",val[0],val[1]);     
      break;

// "N" = Number of digital patterns for trigger mode (was 6)
// "Nx" - x number of digital patterns will be used (0->12
// Note: In triggered mode controller wraps around
    case 'N':
      seqNo = (int) val[0];    
      if ((seqNo >= 0) && (seqNo < SEQUENCELENGTH)) patternLength = seqNo;
      sprintf(outBuf,"N,%d\r\n",val[0]);         
      break;

// "K" = sKip triggers (was 7)
// "Kx" - x number of events to skip on input pin
    case 'K':
      skipTriggers = int(val[0]);
      sprintf(outBuf,"K,%d\r\n",val[0]);     
      break;

// "R" = Run trigger mode (was 8)
// Note: Trigger mode overrides blanking mode (if it was active)??
    case 'R':
      if (patternLength > 0) {
        sequenceNr = 0;
        triggerNr = -skipTriggers;
        triggerState = digitalRead(inPin);
        writeZeros();
        triggerMode = true;
      }
      sprintf(outBuf,"R\r\n"); 
      break;

// "E" = End trigger Mode (was 9)
// duplex - RETURN "E,x"+crlf - x is number of triggers on the last run
// Should always return "x"+crlf
    case 'E':
      triggerMode = false;
      writeZeros();
      sprintf(outBuf,"E,%d\r\n",triggerNr);
      resp=true;  
      break;

// "T" = Time intervals for trigger mode (was 10)
// "Txtt" - x is interval number index (0 -> 12)
//        - tt is interval (in ms) unsigned int format.
    case 'T':
      seqNo = (int) val[0];  
      if ((seqNo >= 0) && (seqNo < SEQUENCELENGTH)) {
        triggerDelay[seqNo] = (int) val[1];
      }
      sprintf(outBuf,"T,%d\r\n",val[0]); 
      break;

// "I" = Iterations of trigger patterns (was 11)
// "Ix" - x number of iterations in timed trigger mode
    case 'I':
      repeatPattern = (int) val[0];
      sprintf(outBuf,"I,%d\r\n",val[0]); 
      break;

/////////////////////////////////////////////////////////////////////////////      
// NOTE: if "G" command has been received we sit in this loop repeating 
// patterns until we reach patternLength or "Serial.available" 
// "G" = Go timed trigger mode.. (was 12)
// Vout patterns output (LEDs on and off) as below
// Pre-Sets.......:
//  - Patterns are set using "T" command
//  - Intervals are set with "D" command
//  - Number of Patterns set with "N" command
//  - Iterations are set with "I" command
// duplex - RETURN "G"+crlf
    case 'G':
      if (patternLength > 0) {
        writeZeros();
        for (int i = 0; i < repeatPattern && (Serial.available() == 0); i++) {
          for (int j = 0; j < patternLength && (Serial.available() == 0); j++) {
            writePattern(triggerPattern[j]);
            delay(triggerDelay[j]);
          }
        }
        writeZeros();
      }
      sprintf(outBuf,"G\r\n"); 
      break;
/////////////////////////////////////////////////////////////////////////////

// "L" = logic in on Analogue Pins (was 40)
// "L,x" - x = Analogue Pin number 0-5 
//      If x = 6  this means "ALL" - perhaps && channels?
//      I'm just going to send back 1! - don't understand "ALL"
// RETURN "L,1/0"+crlf  
    case 'L':
      chan = val[0];
      tmp=1;
      if ((chan>= 0) && (chan < nADC)) tmp = digitalRead(ADCpin[chan]); 
      sprintf(outBuf,"L,%d\r\n", tmp);
      resp=true;
      break;

// "A" = Analogue read pin value (was 41)
// "A,x" - x = Analogue pin number (0->5)
// Duplex - RETURN "A,v"+crlf - where v is 10-bit value on AnaloguePin(x)
// Always return "v"+crlf
    case 'A':
      chan = (int) val[0];
      tmp=0;
      if ((chan >= 0) && (chan < nADC)) tmp = analogRead(ADCpin[chan]);
      sprintf(outBuf,"A,%d\r\n", tmp);
      resp=true;
      break;      
 
// "D" set input pullup (was 42)
// "D,chan,state" if state true set "Input_PullUp"
// duplex - RETURN "D"+crlf
    case 'D':
      chan = val[0];
      state = (int) val[1];  
      if ((chan >= 0) && (chan < nADC)) {
      if  (state) pinMode(ADCpin[chan], INPUT_PULLUP);
      else pinMode(ADCpin[chan], INPUT);
      } 
      sprintf(outBuf,"D,%d,%d\r\n",val[0],val[1]); 
      break;

//////////////////  XYZ Stage control Commands  //////////////////      
// Code adapted from PIEZO CONCEPT X,Y,Z
// "U,chan" get axis range for x,y,z
// RETURN "U,100"+crlf {200um in x,y, 100um in z}?
    case 'U':
      chan= (int) val[0];
      switch (chan) {
        case 0:
          sprintf(outBuf,"U,%d\r\n", xRange);
          resp=true;
          break;
        case 1:
          sprintf(outBuf,"U,%d\r\n", yRange);
          resp=true;          
          break;
        case 2:
          sprintf(outBuf,"U,%d\r\n", zRange);
          resp=true;
          break;            
        }
      break;

    case 'X': 
      posn = val[0];
      if (posn < 0) posn = 0;
      if (posn > xRange) posn = xRange;
      xOut = posn/xRange * (1 << xPWMres);      
      ledcWrite(xChan, xOut); 
      sprintf(outBuf,"X\r\n");
      resp=true;
      break;
      
    case 'Y':
      posn = val[0];
      if (posn < 0) posn = 0;
      if (posn > yRange) posn = yRange;
      yOut = posn/yRange * (1 << yPWMres);  
      ledcWrite(yChan, yOut); 
      sprintf(outBuf,"Y\r\n");
      resp=true;
      break;

    case 'Z':
      posn = val[0];
      if (posn < 0) posn = 0;
      if (posn > zRange) posn = zRange;
      zOut = posn/zRange * (1 << zPWMres);  
      ledcWrite(zChan, zOut); 
      sprintf(outBuf,"Z\r\n");
      resp=true;
      break;

// Stage and Focus Sequencing... this is not yet coded on the MMAdapter-side
// "M, chan, seqNo, posn"      
     case 'M':
      seqNo = (int) val[1];
      if ((seqNo >= 0) && (seqNo < SEQUENCELENGTH)) { 
        posn = val[2];
        if (posn < 0) posn = 0;
        
        chan= (int) val[0];
        switch (chan) {
          case 0:
            if (posn > xRange) posn = xRange;
            xOut = posn/xRange * (1 << xPWMres);
            xPosn[seqNo] = xOut;
            break;
          case 1:
            if (posn > yRange) posn = yRange;
            yOut = posn/yRange * (1 << yPWMres);
            yPosn[seqNo] = yOut;
            break;
          case 2:
            if (posn > zRange) posn = zRange;
            zOut = posn/zRange * (1 << zPWMres);
            zPosn[seqNo] = zOut;
            break;            
        }
      }
      sprintf(outBuf,"M,%d,%d,%3.2f\r\n",val[0],val[1],val[2]);
      break;

  }

return (resp);
}

// Service functions....
// **************
void writeZeros() {
// **************
// self-explanatory
  for (int i = 0; i < nDAC; i++) {
      ledcWrite(i, 0);
  }
}

// *****************************
void writePattern(byte pattern) {
// *****************************
// self-explanatory
  for (int i = 0; i < nDAC; i++) {
      if (bitRead(pattern, i) == 0) ledcWrite(i, 0);
      else ledcWrite(i, Vout[i]);
  }
}

//============================
void writeln(String txt) {
//============================
// Write to the OLED screen so we can watch comms traffic etc
// most of this code is just to scroll the lines up!
// There's an Adafruit command to do it..BUT
// Here's some Q&D grunt code!

  for (int i = 0; i < (linemax - 1); i++) {
    txtlines[i] = txtlines[i + 1];  // shuffle the txt up!
  }

  txtlines[linemax - 1] = txt;

  display.clearDisplay();

  for (int i = 0; i < linemax; i++) {
    display.setCursor(0, i * 8);  // set line
    txt = txtlines[i];
    if (txt.indexOf("PC:") >= 0) {
      display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);  // Draw Black text, White background
    } else {
      display.setTextColor(SSD1306_WHITE, SSD1306_BLACK);  // Draw White text, Black background
    }
    display.print(txt);
  }
  display.display();  // update display
}