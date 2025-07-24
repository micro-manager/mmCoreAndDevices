// Firmare code for the Teensy 3.5 (and 3.6?) compatible with the
// standard MM Arduino device adapter.
// Based on code by Bonno Medens (which was based on code by Nico Stuurman, who wrote this code)
// Could be extended to support PWM outputs and to support
// sequencing for analog output values.


/*
 * First, a serial command can directly set the digital output patern
 * 
 * Second, a series of patterns can be stored in the arduino and a trigger on pin 2 
 *    wil trigger the consecutive pattern (trigger mode).
 *    
 * Third, intervals between consecutive patterns can be specified and paterns will be
 * generated at these specified time points (timed trigger mode).
 */

/* Interface specifications
 *  Digital pattern specification: single byte, bit 0 corresponds to pin 8,
 *    bit 1 to pin 9, etc.. Bits 7 and 8 will not be used (and should stay 0).
 */

/* Set digital output command: 1p
 *  Where p is the desired digital pattern. 
 *  
 *  Controller will return 1 to indicate succesfull execution.
 */

/* Get Digital output command: 2
 *  
 *  Controller will return 2p. Where p is the current digital output pattern
 */

/* Digital output command 3xvv
 *    Where x is the ouput channel and vv is the output in
 *    a 12-bit significant number.
 *    
 *    Controller will return 3xvv
 */

/* Get analogue output: 3
  */

/* Set digital pattern for triggerd mode: 5xd
 *    Where x is the number of the pattern (currrently, 12 patterns can be stored).
 *    and d is the digital pattern to be stored at that position. Note that x should be
 *    real number (i.e., not ASCI encoded). See 50 for a faster way to load sequences.
 *    
 *    Controller will return 5xd
 */

/* Set the Number of digital patterns to be used: 6x
 *   Where x indicates how many digital patterns will be used (up to SEQUENCEMAXLENGTH
 *   patterns maximum).  In triggered mode, after reaching this many triggers, 
 *   the controller will re-start the sequence with the first pattern.
 *   
 *   Controller will return 6x
*/

/* Skip tirgger: 7x
 *    Where x indicates how many digital change events on the trigger input pin
 *    will be ignored.
 *    
 *    Controller will respond with 7x
 */

/* Start trigger mode: 8 
 *    Controller will return 8 to indicate start of trigger mode
 *    Stop triggered at 9. Trigger mode will supersede (but not stop)
 *    blanking mode (if it was active)
 */

/* Stop Trigger mode: 9
 *    Controller will return 9x where x is the number of triggers received during the last
 *    trigger mode run
 */

/* Set time interval for timed trigger mode: 10xtt
 *    Where x is the number interval (currently 12 intervals can be stored)
 *    and tt is the interval (in ms) in Arduino unsigned int format.
 *    
 *    Controller will return 11x
 */

/* Sets how often the timed pattern will be repeated 11x
 *    This value will be used in timed-trigger mode and sets how often the ouput
 *    pattern will be repeated.
 *    
 *    Controller will return 11x
 */

/* Starts timed trigger mode: 12
 *    In timed trigger mode, digital patterns as with function 5 will appear on the
 *    output pins with intervals (in ms) as set with function 10. After the number of 
 *    patterns set with function 6, the pattern will be repeated for the number of times
 *    set with function 11. Any input character (which will be processed) will stop
 *    the pattern generation
 *    
 *    Controller will return 12.
 */

/* Start blanking mode: 20
 *    In blanking mode, zeroes will be written on the output pins when the trigger pin
 *    is low, when the trigger pin is high, the pattern set with command 1 will be
 *    applied to the output pins.
 *    
 *    Controller will return 20
 */

/* Stop blanking mode: 21
 *    Stopts blanking mode
 *    
 *    Controller returns 21
 */

/* Get Identifcatio: 30
 *     Returns in ASCI MM-Ard\r\n
 */

/* Get Version: 31
 *    Returns: version number in ASCI \r\n
 */

/* Get Max number of patterns that can be uploaded: 32
 *   Returns: Max number of patterns as an unsigned int, 2 bytes, highbyte first
 *   Available as of version 3
 */

/** Fast version to upload a digital sequence: 33
  * first 2 bytes indicate the number of bytes to be uploaded (and length of the sequence)
  * followed by the indicated number of bytes
  * Available as of version 4
 */


/* Read digital state of analog input pins 0-5: 40
 *    Returns raw value of PINC
 */

/* Read analog state of input pins 0-5: 41x
 *    x=0-5. Returns analog value as a 10-bit number (0-1023)
 */

/*
 * Possible extensions:
 *   Set and Get Mode (low, change, rising, falling) for trigger mode
 *   Get digital patterm
 *   Get Number of digital patterns
 */
// pin on whick to receive the trigger (2 and 3 can be used with interrupts, although this code does not use them)
// to read out the state of inPin_ faster, ise
// int inPinBit_ = 1<< inPin_; // bit mask


unsigned int version_ = 4;  // Change from 3 > 4: Make SequenceLength Configurable.

const int SEQUENCELENGTH = 10000;  // This number can be increased until you run out of memory
byte triggerPattern_[SEQUENCELENGTH];
unsigned int triggerDelay_[SEQUENCELENGTH];
uint32_t patternLength_ = 0;
byte repeatPattern_ = 0;
volatile uint32_t triggerNr_;   // total # of triggers in this run (0-based)
volatile uint32_t sequenceNr_;  // # of trigger in sequence (0-based)
int skipTriggers_ = 0;          // # of triggers to skip before starting to generate patterns
byte currentPattern_ = 0;
const unsigned long timeOut_ = 1000;
bool blanking_ = false;
bool blankOnHigh_ = false;
bool triggerMode_ = false;
boolean triggerState_ = false;

// New additions for rewrite code
byte realCurrentPattern_ = 0;
byte pindAlt = 0;

// New additions for use of internal DAC
// Default Channel power
const unsigned int ch1Power = 4095;
const unsigned int ch2Power = 4095;
const unsigned int ch3Power = 4095;
const unsigned int ch4Power = 4095;
const unsigned int ch5Power = 4095;
const unsigned int ch6Power = 4095;
const unsigned int ch7Power = 4095;
const unsigned int ch8Power = 4095;
unsigned int msblsb[] = { ch1Power, ch2Power, ch3Power, ch4Power, ch5Power, ch6Power, ch7Power, ch8Power };

const float freq = 14648.437;
const unsigned int PWMresolution = 12;



// Channel selection
const uint8_t inPin_ = 2;
const uint8_t ao1 = A21;  // Non Dac blanking -> const int ch1 = 8;
const uint8_t ao2 = A22;  // Non Dac blanking -> const int ch2 = 9;
const uint8_t ao3 = 3;
const uint8_t ao4 = 4;
const uint8_t ao5 = 5;
const uint8_t ao6 = 6;
const uint8_t ao7 = 7;
const uint8_t ao8 = 8;

const uint8_t aos[8] = { A21, A22, 3, 4, 5, 6, 7, 8 };

const uint8_t dos[8] = { 13, 14, 15, 16, 17, 18, 19, 20 };

void setup() {
  // put your setup code here, to run once:
  Serial.begin(500000);  //Baud rate

  pinMode(inPin_, INPUT);

  //pinMode(ch1, OUTPUT);     is disabled with DAC blanking for SAMD boards
  //pinMode(ch2, OUTPUT);     is disabled with DAC blanking for SAMD boards

  for (uint8_t i = 2; i < 8; i++) {
    pinMode(aos[i], OUTPUT);
    analogWrite(aos[i], 0);
  }

  for (uint8_t i = 0; i < 8; i++) {
    pinMode(dos[i], OUTPUT);
    digitalWriteFast(dos[i], LOW);
  }

  // New for PWM for more information https://www.pjrc.com/teensy/td_pulse.html
  analogWriteResolution(PWMresolution);  // Sets PWM resolution to 12 bits
  analogWriteFrequency(3, freq);         // Sets PWM frequency of pins 3 and 4
  analogWriteFrequency(5, freq);         // Sets PWM frequency of pins 5 and 6
  analogWriteFrequency(7, freq);         // Sets PWM frequency of pins 7 and 8

  for (unsigned int i = 0; i < SEQUENCELENGTH; i++) {
    triggerPattern_[i] = 0;
    triggerDelay_[i] = 0;
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    int inByte = Serial.read();
    switch (inByte) {
      // Set digital output
      case 1:
        if (waitForSerial(timeOut_)) {
          currentPattern_ = Serial.read();
          if (!blanking_) {
            realCurrentPattern_ = currentPattern_;
            writeDigitalPattern(currentPattern_);
          }
          Serial.write(byte(1));
        }
        break;

      // Get digital output
      case 2:
        Serial.write(byte(2));
        Serial.write(realCurrentPattern_);
        break;


        // Set Analogue output (TODO: save for 'Get Analogue output')
      case 3:
        if (waitForSerial(timeOut_)) {
          uint8_t channel = Serial.read();
          if (waitForSerial(timeOut_)) {
            byte msb = Serial.read();
            msb &= B00001111;
            if (waitForSerial(timeOut_)) {
              byte lsb = Serial.read();
              msblsb[channel] = (int)lsb + (int)msb * 256;  // Added DAC blanking
              analogWrite(aos[channel], msblsb[channel]);
              Serial.write(byte(3));
              Serial.write(channel);
              Serial.write(msb);
              Serial.write(lsb);
            }
          }
        }
        break;

        // Sets the specified digital pattern for triggered mode
      case 5:
        if (waitForSerial(timeOut_)) {
          int patternNumber = Serial.read();
          if ((patternNumber >= 0) && (patternNumber < SEQUENCELENGTH)) {
            if (waitForSerial(timeOut_)) {
              triggerPattern_[patternNumber] = Serial.read();
              // triggerPattern_[patternNumber] = triggerPattern_[patternNumber] & B00111111; Removed for extra channels
              Serial.write(byte(5));
              Serial.write(patternNumber);
              Serial.write(triggerPattern_[patternNumber]);
              break;
            }
          }
        }
        Serial.write("n:");  //Serial.print("n:");
        break;

      // Sets the number of digital patterns that will be used
      case 6:
        if (waitForSerial(timeOut_)) {
          int pL = Serial.read();
          if ((pL >= 0) && (pL <= SEQUENCELENGTH)) {
            patternLength_ = pL;
            Serial.write(byte(6));
            Serial.write(patternLength_);
          }
        }
        break;

      // Skip triggers
      case 7:
        if (waitForSerial(timeOut_)) {
          skipTriggers_ = Serial.read();
          Serial.write(byte(7));
          Serial.write(skipTriggers_);
        }
        break;

      //  starts trigger mode
      case 8:
        if (patternLength_ > 0) {
          sequenceNr_ = 0;
          triggerNr_ = -skipTriggers_;
          triggerState_ = digitalRead(inPin_) == HIGH;
          realCurrentPattern_ = 0;
          writeDigitalPattern(0);
          Serial.write(byte(8));
          triggerMode_ = true;
        }
        break;

        // return result from last triggermode
      case 9:
        triggerMode_ = false;
        realCurrentPattern_ = 0;
        writeDigitalPattern(0);
        Serial.write(byte(9));
        Serial.write(triggerNr_);
        break;

      // Sets time interval for timed trigger mode
      // Tricky part is that we are getting an unsigned int as two bytes
      case 10:
        if (waitForSerial(timeOut_)) {
          int patternNumber = Serial.read();
          if ((patternNumber >= 0) && (patternNumber < SEQUENCELENGTH)) {
            if (waitForSerial(timeOut_)) {
              unsigned int highByte = 0;
              unsigned int lowByte = 0;
              highByte = Serial.read();
              if (waitForSerial(timeOut_))
                lowByte = Serial.read();
              highByte = highByte << 8;
              triggerDelay_[patternNumber] = highByte | lowByte;
              Serial.write(byte(10));
              Serial.write(patternNumber);
              break;
            }
          }
        }
        break;

      // Sets the number of times the patterns is repeated in timed trigger mode
      case 11:
        if (waitForSerial(timeOut_)) {
          repeatPattern_ = Serial.read();
          Serial.write(byte(11));
          Serial.write(repeatPattern_);
        }
        break;

      //  starts timed trigger mode
      case 12:
        if (patternLength_ > 0) {
          realCurrentPattern_ = 0;
          writeDigitalPattern(0);
          Serial.write(byte(12));
          for (byte i = 0; i < repeatPattern_ && (Serial.available() == 0); i++) {
            for (uint32_t j = 0; j < patternLength_ && (Serial.available() == 0); j++) {
              realCurrentPattern_ = triggerPattern_[j];
              writeDigitalPattern(triggerPattern_[j]);
              delay(triggerDelay_[j]);
            }
          }
          realCurrentPattern_ = 0;
          writeDigitalPattern(0);
        }
        break;

      // Blanks output based on TTL input
      case 20:
        blanking_ = true;
        Serial.write(byte(20));
        break;

      // Stops blanking mode
      case 21:
        blanking_ = false;
        Serial.write(byte(21));
        break;

      // Sets 'polarity' of input TTL for blanking mode
      case 22:
        if (waitForSerial(timeOut_)) {
          int mode = Serial.read();
          if (mode == 0)
            blankOnHigh_ = true;
          else
            blankOnHigh_ = false;
        }
        Serial.write(byte(22));
        break;

      // Gives identification of the device
      case 30:
        Serial.println("MM-Ard-1");
        break;

      // Returns version string
      case 31:
        Serial.println(version_);
        break;

      // returns Maximum number of patterns for sequencing
      case 32:
        Serial.write(byte(32));
        Serial.write(highByte(SEQUENCELENGTH));
        Serial.write(lowByte(SEQUENCELENGTH));
        break;

      // Faster way of uploading sequence:
      case 33:
        {
          unsigned int highByte = 0;
          unsigned int lowByte = 0;
          unsigned int count = 0;
          if (waitForSerial(timeOut_)) {
            highByte = Serial.read();
            if (waitForSerial(timeOut_)) {
              lowByte = Serial.read();
              highByte = highByte << 8;
              unsigned int expectedNumPatterns = highByte | lowByte;
              if ((expectedNumPatterns >= 0) && (expectedNumPatterns < SEQUENCELENGTH)) {
                while (count < expectedNumPatterns && waitForSerial(timeOut_)) {
                  triggerPattern_[count] = Serial.read();
                  count++;
                }
              }
            }
          }
          patternLength_ = count;
          Serial.write(byte(33));
          Serial.write(highByte(count));
          Serial.write(lowByte(count));
        }
        break;



      //Not implemented
      case 40:
        Serial.write(byte(40));
        //Serial.write( PINC);
        Serial.write(byte(0));
        break;

      case 41:
        if (waitForSerial(timeOut_)) {
          int pin = Serial.read();
          if (pin >= 0 && pin <= 5) {
            int val = analogRead(pin);
            Serial.write(byte(41));
            Serial.write(pin);
            Serial.write(highByte(val));
            Serial.write(lowByte(val));
          }
        }
        break;

      case 42:
        if (waitForSerial(timeOut_)) {
          int pin = Serial.read();
          if (waitForSerial(timeOut_)) {
            int state = Serial.read();
            Serial.write(byte(42));
            Serial.write(pin);
            if (state == 0) {
              digitalWrite(14 + pin, LOW);
              Serial.write(byte(0));
            }
            if (state == 1) {
              digitalWrite(14 + pin, HIGH);
              Serial.write(byte(1));
            }
          }
        }
        break;
    }
  }

  // In trigger mode, we will blank even if blanking is not on..
  if (triggerMode_) {
    pindAlt = digitalReadFast(inPin_);
    boolean tmp = pindAlt;
    if (tmp != triggerState_) {
      if (blankOnHigh_ && tmp) {
        realCurrentPattern_ = 0;
        writeDigitalPattern(0);
      } else if (!blankOnHigh_ && !tmp) {
        realCurrentPattern_ = 0;
        writeDigitalPattern(0);
      } else {
        if (triggerNr_ >= 0) {
          realCurrentPattern_ = triggerPattern_[sequenceNr_];
          writeDigitalPattern(triggerPattern_[sequenceNr_]);
          sequenceNr_++;
          if (sequenceNr_ >= patternLength_)
            sequenceNr_ = 0;
        }
        triggerNr_++;
      }

      triggerState_ = tmp;
    }
  } else if (blanking_) {
    if (blankOnHigh_) {
      if (!digitalReadFast(inPin_)) {
        realCurrentPattern_ = currentPattern_;
        writeDigitalPattern(currentPattern_);
      } else {
        realCurrentPattern_ = 0;
        writeDigitalPattern(0);
      }
    } else {
      if (!digitalReadFast(inPin_)) {
        realCurrentPattern_ = 0;
        writeDigitalPattern(0);
      } else {
        realCurrentPattern_ = currentPattern_;
        writeDigitalPattern(currentPattern_);
      }
    }
  }
}

bool waitForSerial(unsigned long timeOut) {
  unsigned long startTime = millis();
  while (Serial.available() == 0 && (millis() - startTime < timeOut)) {}
  if (Serial.available() > 0)
    return true;
  return false;
}


void writeDigitalPattern(byte pattern) {
  for (uint8_t i = 0; i < 8; i++) {
    digitalWriteFast(dos[i], bitRead(pattern, i));
  }
}

