/*
 * This goal of the application is to set the digital output on pins 8-13 
 * This can be accomplished in three ways.  First, a serial command can directly set
 * the digital output pattern.  Second, a series of patterns can be stored in the 
 * Arduino and TTLs coming in on pin 2 will then trigger to the consecutive pattern (trigger mode).
 * Third, intervals between consecutive patterns can be specified and paterns will be 
 * generated at these specified time points (timed trigger mode).
 *
 * Interface specifications:
 * digital pattern specification: single byte, bit 0 corresponds to pin 8, 
 *   bit 1 to pin 9, etc..  Bits 7 and 8 will not be used (and should stay 0).
 *
 * Set digital output command: 1p
 *   Where p is the desired digital pattern.  Controller will return 1 to 
 *   indicate succesfull execution.
 *
 * Get digital output command: 2
 *   Controller will return 2p.  Where p is the current digital output pattern
 *
 * Set Analogue output command: 3xvv
 *   Where x is the output channel (either 1 or 2), and vv is the output in a 
 *   12-bit significant number.
 *   Controller will return 3xvv:
 *
 * Get Analogue output:  4
 *
 *
 * Set digital patten for triggered mode: 5xd 
 *   Where x is the number of the pattern (Quesry the max number of patterns with 32, default is 12, but number can be changed in the firmware).
 *   and d is the digital pattern to be stored at that position.  Note that x should
 *   be the real number (i.e., not  ASCI encoded)
 *   Controller will return 5xd 
 *
 * Set the Number of digital patterns to be used: 6x
 *   Where x indicates how many digital patterns will be used (currently, up to 12
 *   patterns maximum).  In triggered mode, after reaching this many triggers, 
 *   the controller will re-start the sequence with the first pattern.
 *   Controller will return 6x
 *
 * Skip trigger: 7x
 *   Where x indicates how many digital change events on the trigger input pin
 *   will be ignored.
 *   Controller will respond with 7x
 *
 * Start trigger mode: 8
 *   Controller will return 8 to indicate start of triggered mode
 *   Stop triggered a 9. Trigger mode will  supersede (but not stop) 
 *   blanking mode (if it was active)
 * 
 * Stop Trigger mode: 9
 *   Controller will return 9x where x is the number of triggers received during the last
 *   trigger mode run
 *
 * Set time interval for timed trigger mode: 10xtt
 *   Where x is the number of the interval (currently, 12 intervals can be stored)
 *   and tt is the interval (in ms) in Arduino unsigned int format.  
 *   Controller will return 10x
 *
  * Sets how often the timed pattern will be repeated: 11x
 *   This value will be used in timed-trigger mode and sets how often the output
 *   pattern will be repeated. 
 *   Controller will return 11x
 *  
 * Starts timed trigger mode: 12
 *   In timed trigger mode, digital patterns as set with function 5 will appear on the 
 *   output pins with intervals (in ms) as set with function 10.  After the number of 
 *   patterns set with function 6, the pattern will be repeated for the number of times
 *   set with function 11.  Any input character (which will be processed) will stop 
 *   the pattern generation.
 *   Controller will retun 12.
 * 
 * Start blanking Mode: 20
 *   In blanking mode, zeroes will be written on the output pins when the trigger pin
 *   is low, when the trigger pin is high, the pattern set with command #1 will be 
 *   applied to the output pins. 
 *   Controller will return 20
 *
 * Stop blanking Mode: 21
 *   Stops blanking mode.  Controller returns 21
 *
 * Blanking mode trigger direction: 22x
 *   Sets whether to blank on trigger high or trigger low.  x=0: blank on trigger high,
 *   x=1: blank on trigger low.  x=0 is the default
 *   Controller returns 22
 *
 * 
 * Get Identification: 30
 *   Returns (asci!) MM-Ard\r\n
 *
 * Get Version: 31
 *   Returns: version number (as ASCI string) \r\n
 *
 * Get Max number of patterns that can be uploaded: 32
 *   Returns: Max number of patterns as an unsigned int, 2 bytes, highbyte first
 *   Available as of version 3
 *
 * Fast version to upload a digital sequence: 33
 *   first 2 bytes indicate the number of bytes to be uploaded (and length of the sequence)
 *   followed by the indicated number of bytes
 *   Available as of version 4
 *
 * Get DA channel count: 34
 *   Returns: 34 followed by 1 byte with the number of DA channels
 *   Available as of version 5
 *
 * Get digital pin count: 35
 *   Returns: 35 followed by 1 byte with the number of digital output pins
 *   Available as of version 5
 *
 * Get DA channel voltage range and resolution: 36x
 *   Where x is the DA channel (0-based).
 *   Returns: 36 followed by channel, then min voltage, max voltage, and the
 *   maximum digital code for that channel:
 *     - min voltage: signed long (int32_t), microvolts, 4 bytes, highbyte first
 *     - max voltage: signed long (int32_t), microvolts, 4 bytes, highbyte first
 *     - max digital code: unsigned long (uint32_t), 4 bytes, highbyte first
 *       (e.g. 4095 for a 12-bit DAC channel)
 *   A min/max voltage of 0/0 means the voltage range is not known.
 *   Available as of version 6
 *
 * Get max number of DA sequence events per channel: 37
 *   Returns: 37 followed by the max number of events per channel, unsigned int, 2 bytes, highbyte first
 *   Available as of version 6
 *
 * Upload a DA voltage sequence for one channel: 38xccvv...
 *   Where x is the DA channel (0-based), cc is the number of events (unsigned int, 2 bytes,
 *   highbyte first), and vv... is cc pairs of bytes, each pair a 12-bit significant number
 *   (msb, lsb) as in command 3.
 *   Controller returns 38, channel, then the number of events actually stored (2 bytes,
 *   highbyte first) - a value less than cc means the upload was truncated (e.g. timed out).
 *   Available as of version 6
 *
 * Start DA (analog) triggered sequence output: 39
 *   Every DA channel with a non-empty uploaded sequence advances to its next value on each
 *   transition (rising or falling) of the trigger input pin (the same pin used for digital
 *   triggered mode, command 8). All channels share one step index, so channels with
 *   sequences of equal length stay in lock-step. Controller returns 39.
 *   Available as of version 6
 *
 * Stop DA (analog) triggered sequence output: 43
 *   Controller returns 43x where x is the number of triggers received during the last DA
 *   sequence run. Output is left at its last value (not forced to zero).
 *   Available as of version 6
 *
 *
 * Read digital state of analogue input pins 0-5: 40
 *   Returns raw value of PINC (two high bits are not used)
 *
 * Read analogue state of pint pins 0-5: 41x
 *   x=0-5.  Returns analogue value as a 10-bit number (0-1023)
 *
 *
 * 
 * Possible extensions:
 *   Set and Get Mode (low, change, rising, falling) for trigger mode
 *   Get digital patterm
 *   Get Number of digital patterns
 */


 /************* For DA accessory chips, edit this section ********/

// If you have one of these DA chips attached, uncomment the appropriate define.
// WARNING: if none of these are defined, numDAChannels_ below becomes 0 and every
// DA channel silently does nothing - single-value writes (command 3) and sequence
// uploads (command 38) will report "success" or "0 events stored" respectively,
// with no error, even though the host may already have "Volts"/"MaxVolt"/"Sequence"
// properties configured for DA channels from a previous session. Double-check this
// matches your attached hardware before troubleshooting anything else DA-related.
// The same applies when a DAC is compiled in but fails to initialize at runtime;
// command 34 then reports 0 channels (see availableDAChannels() below), so the
// host refuses to initialize DA devices instead of writing into a void.
// #define TLV5618
// #define TLV56x8
// #define MCP4728

// Voltage range and resolution of the DA channels, reported to the host via
// command 36.  Currently the same fixed values are used for all channels;
// change these if the attached DAC hardware differs.
const int32_t DA_MIN_VOLTAGE_MICROV = 0;          // 0 V
const int32_t DA_MAX_VOLTAGE_MICROV = 5000000L;   // 5 V
const uint32_t DA_NUM_STEPS = 4095;               // max digital code (12-bit)

/**************** End editing DA accessory chips section ********/

const unsigned int version_ = 6;

#ifdef MCP4728
#include <Wire.h>
#include <Adafruit_MCP4728.h>
#include <EEPROM.h>
#endif

 
#if defined TLV5618
const uint8_t numDAChannels_ = 2;
#elif defined TLV56x8
const uint8_t numDAChannels_ = 4;
#elif defined MCP4728
const uint8_t numDAChannels_ = 4;
#else
const uint8_t numDAChannels_ = 0;
#endif


#ifdef MCP4728
static const uint8_t MCP4728_ADDR   = 0x60;
static const uint8_t LDAC_PIN       = 4;
static const uint8_t RDY_PIN        = 3;
Adafruit_MCP4728 mcp;
bool mcp_ok = false;
#endif

// Number of DA channels actually usable right now.  This is the compile-time
// numDAChannels_, except that a DAC which failed to initialize (MCP4728 not
// found on the I2C bus) has no usable channels: analogueOut() discards every
// write to it, so reporting the compile-time count to the host would make it
// create DA devices whose writes are silently dropped.
uint8_t availableDAChannels() {
#ifdef MCP4728
  if (!mcp_ok) return 0;
#endif
  return numDAChannels_;
}

const uint8_t numDigitalPins_ = 6;
   
   // pin on which to receive the trigger (2 and 3 can be used with interrupts, although this code does not use interrupts)
   int inPin_ = 2;
   // to read out the state of inPin_ faster, use 
   int inPinBit_ = 1 << inPin_;  // bit mask 
   
   // pin connected to DIN of TLV5618
   #if defined TLV5618 || defined TLV56x8
   int dataPin = 3;
   // pin connected to SCLK of TLV5618
   int clockPin = 4;
   #endif
   // pin connected to CS of TLV5618
   #ifdef TLV5618
   int latchPin = 5;
   #endif

   #ifdef TLV56x8
   int CS1 = 5;  // Used for TLV56X8
   int CS2 = 6;  // Used for TLV56X8
   #endif

   const uint16_t SEQUENCELENGTH = 256;  // Can be increased, but pay attention that there is significant memory left for local variables
   byte triggerPattern_[SEQUENCELENGTH]; 
   unsigned int triggerDelay_[SEQUENCELENGTH]; 
   int patternLength_ = 0;
   byte repeatPattern_ = 0;
   volatile long triggerNr_; // total # of triggers in this run (0-based)
   volatile long sequenceNr_; // # of trigger in sequence (0-based)
   int skipTriggers_ = 0;  // # of triggers to skip before starting to generate patterns
   byte currentPattern_ = 0;
   const unsigned long timeOut_ = 1000;
   bool blanking_ = false;
   bool blankOnHigh_ = false;
   bool triggerMode_ = false;
   boolean triggerState_ = false;

   // Analog (DA) voltage sequence support - mirrors the digital pattern sequence above, but
   // stores a 12-bit code sequence per DA channel and is driven off the same trigger pin
   // (inPin_) via a single shared step index (see command 39's doc comment).
   const uint8_t MAX_DA_CHANNELS_ = 4;      // largest numDAChannels_ across all chip branches
   const uint16_t DA_SEQUENCELENGTH = 32;   // per channel; increase with care - see SEQUENCELENGTH comment above
   uint16_t daSequence_[MAX_DA_CHANNELS_][DA_SEQUENCELENGTH];
   uint16_t daSequenceLength_[MAX_DA_CHANNELS_] = {0, 0, 0, 0};
   volatile long daTriggerNr_;    // total # of triggers in this DA-sequence run
   volatile long daSequenceNr_;   // shared step index into every channel's daSequence_[]
   bool daTriggerMode_ = false;
   boolean daTriggerState_ = false;

 void setup() {
   // Higher speeds do not appear to be reliable
   Serial.begin(57600);
  
   pinMode(inPin_, INPUT);
   #if defined TLV5618 || defined TLV56x8
   pinMode (dataPin, OUTPUT);
   pinMode (clockPin, OUTPUT);
   #endif
   #ifdef TLV5618
   pinMode (latchPin, OUTPUT);
   #endif
   #ifdef TLV56x8
   pinMode (CS1, OUTPUT);
   pinMode (CS2, OUTPUT);
   #endif
   pinMode(8, OUTPUT);
   pinMode(9, OUTPUT);
   pinMode(10, OUTPUT);
   pinMode(11, OUTPUT);
   pinMode(12, OUTPUT);
   pinMode(13, OUTPUT);
   
   // Set analogue pins as input:
   DDRC = DDRC & B11000000;
   // Turn on build-in pull-up resistors
   PORTC = PORTC | B00111111;
   
   #ifdef TLV5618
   digitalWrite(latchPin, HIGH);   
   #endif
   #ifdef TLV56x8
   digitalWrite(CS1, HIGH);
   digitalWrite(CS2, HIGH);
   #endif

   #ifdef MCP4728
   pinMode(LDAC_PIN, OUTPUT);
   digitalWrite(LDAC_PIN, LOW);
   pinMode(RDY_PIN, INPUT_PULLUP);
   Wire.begin();
   mcp_ok = mcp.begin(MCP4728_ADDR);
   {
     const int INIT_FLAG_ADDR = 0;
     const byte INIT_DONE = 0xA5;
     if (mcp_ok) {
       byte initFlag = EEPROM.read(INIT_FLAG_ADDR);
       if (initFlag != INIT_DONE) {
         mcp.fastWrite(0, 0, 0, 0);
         if (mcp.saveToEEPROM())
           EEPROM.update(INIT_FLAG_ADDR, INIT_DONE);
       } else {
         mcp.fastWrite(0, 0, 0, 0);
       }
     }
   }
   #endif

   for (unsigned int i = 0; i < SEQUENCELENGTH; i++) {
      triggerPattern_[i] = 0;
      triggerDelay_[i] = 0;
   }
 }
 
 void loop() {
   if (Serial.available() > 0) {
     int inByte = Serial.read();
     switch (inByte) {
       
       // Set digital output
       case 1 :
          if (waitForSerial(timeOut_)) {
            currentPattern_ = Serial.read();
            // Do not set bits 6 and 7 (not sure if this is needed..)
            currentPattern_ = currentPattern_ & B00111111;
            if (!blanking_)
              PORTB = currentPattern_;
            Serial.write( byte(1));
          }
          break;
          
       // Get digital output
       case 2:
          Serial.write( byte(2));
          Serial.write( PORTB);
          break;
          
       // Set Analogue output (TODO: save for 'Get Analogue output')
       case 3:
         if (waitForSerial(timeOut_)) {
           int channel = Serial.read();
           if (waitForSerial(timeOut_)) {
              byte msb = Serial.read();
              msb &= B00001111;
              if (waitForSerial(timeOut_)) {
                byte lsb = Serial.read();
                if (channel >= 0 && channel < availableDAChannels() && channel < MAX_DA_CHANNELS_)
                  analogueOut(channel, msb, lsb);
                Serial.write( byte(3));
                Serial.write( channel);
                Serial.write(msb);
                Serial.write(lsb);
              }
           }
         }
         break;
         
       // Sets the specified digital pattern
       case 5:
          if (waitForSerial(timeOut_)) {
            int patternNumber = Serial.read();
            if ( (patternNumber >= 0) && (patternNumber < SEQUENCELENGTH) ) {
              if (waitForSerial(timeOut_)) {
                triggerPattern_[patternNumber] = Serial.read();
                triggerPattern_[patternNumber] = triggerPattern_[patternNumber] & B00111111;
                Serial.write( byte(5));
                Serial.write( patternNumber);
                Serial.write( triggerPattern_[patternNumber]);
                break;
              }
            }
          }
          Serial.write( "n:");//Serial.print("n:");
          break;
          
       // Sets the number of digital patterns that will be used
       case 6:
         if (waitForSerial(timeOut_)) {
           int pL = Serial.read();
           if ( (pL >= 0) && (pL <= SEQUENCELENGTH) ) {
             patternLength_ = pL;
             Serial.write( byte(6));
             Serial.write( patternLength_);
           }
         }
         break;
         
       // Skip triggers
       case 7:
         if (waitForSerial(timeOut_)) {
           skipTriggers_ = Serial.read();
           Serial.write( byte(7));
           Serial.write( skipTriggers_);
         }
         break;
         
       //  starts trigger mode
       case 8: 
         if (patternLength_ > 0) {
           sequenceNr_ = 0;
           triggerNr_ = -skipTriggers_;
           triggerState_ = digitalRead(inPin_) == HIGH;
           PORTB = B00000000;
           Serial.write( byte(8));
           triggerMode_ = true;           
         }
         break;
         
         // return result from last triggermode
       case 9:
          triggerMode_ = false;
          PORTB = B00000000;
          Serial.write( byte(9));
          Serial.write( triggerNr_);
          break;
          
       // Sets time interval for timed trigger mode
       // Tricky part is that we are getting an unsigned int as two bytes
       case 10:
          if (waitForSerial(timeOut_)) {
            int patternNumber = Serial.read();
            if ( (patternNumber >= 0) && (patternNumber < SEQUENCELENGTH) ) {
              if (waitForSerial(timeOut_)) {
                unsigned int highByte = 0;
                unsigned int lowByte = 0;
                highByte = Serial.read();
                if (waitForSerial(timeOut_))
                  lowByte = Serial.read();
                highByte = highByte << 8;
                triggerDelay_[patternNumber] = highByte | lowByte;
                Serial.write( byte(10));
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
           Serial.write( byte(11));
           Serial.write( repeatPattern_);
         }
         break;

       //  starts timed trigger mode
       case 12: 
         if (patternLength_ > 0) {
           PORTB = B00000000;
           Serial.write( byte(12));
           for (byte i = 0; i < repeatPattern_ && (Serial.available() == 0); i++) {
             for (int j = 0; j < patternLength_ && (Serial.available() == 0); j++) {
               PORTB = triggerPattern_[j];
               delay(triggerDelay_[j]);
             }
           }
           PORTB = B00000000;
         }
         break;

       // Blanks output based on TTL input
       case 20:
         blanking_ = true;
         Serial.write( byte(20));
         break;
         
       // Stops blanking mode
       case 21:
         blanking_ = false;
         Serial.write( byte(21));
         break;
         
       // Sets 'polarity' of input TTL for blanking mode
       case 22: 
         if (waitForSerial(timeOut_)) {
           int mode = Serial.read();
           if (mode==0)
             blankOnHigh_= true;
           else
             blankOnHigh_= false;
         }
         Serial.write( byte(22));
         break;
         
       // Gives identification of the device
       case 30:
         Serial.println("MM-Ard");
         break;
         
       // Returns version string
       case 31:
         Serial.println(version_);
         break;

        // returns Maximum number of patterns for sequencing
       case 32:
         Serial.write( byte(32));
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
                   triggerPattern_[count] = triggerPattern_[count] & B00111111;
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

       // Returns the number of DA channels
       case 34:
         Serial.write(byte(34));
         Serial.write(byte(availableDAChannels()));
         break;

       // Returns the number of digital output pins
       case 35:
         Serial.write(byte(35));
         Serial.write(byte(numDigitalPins_));
         break;

       // Returns the voltage range and max digital code (signed V, unsigned steps) of the given DA channel
       case 36:
         if (waitForSerial(timeOut_)) {
           int channel = Serial.read();
           int32_t minMicroV = 0, maxMicroV = 0;
           uint32_t numSteps = 0;
           getDaVoltageRangeMicroV(channel, minMicroV, maxMicroV, numSteps);
           Serial.write(byte(36));
           Serial.write(byte(channel));
           Serial.write(byte((minMicroV >> 24) & 0xFF));
           Serial.write(byte((minMicroV >> 16) & 0xFF));
           Serial.write(byte((minMicroV >> 8) & 0xFF));
           Serial.write(byte(minMicroV & 0xFF));
           Serial.write(byte((maxMicroV >> 24) & 0xFF));
           Serial.write(byte((maxMicroV >> 16) & 0xFF));
           Serial.write(byte((maxMicroV >> 8) & 0xFF));
           Serial.write(byte(maxMicroV & 0xFF));
           Serial.write(byte((numSteps >> 24) & 0xFF));
           Serial.write(byte((numSteps >> 16) & 0xFF));
           Serial.write(byte((numSteps >> 8) & 0xFF));
           Serial.write(byte(numSteps & 0xFF));
         }
         break;

       // Returns the maximum number of DA sequence events that can be uploaded per channel
       case 37:
         Serial.write(byte(37));
         Serial.write(highByte(DA_SEQUENCELENGTH));
         Serial.write(lowByte(DA_SEQUENCELENGTH));
         break;

       // Uploads a DA voltage sequence for one channel
       case 38:
         if (waitForSerial(timeOut_)) {
           int channel = Serial.read();
           if (waitForSerial(timeOut_)) {
             unsigned int hi = Serial.read();
             if (waitForSerial(timeOut_)) {
               unsigned int lo = Serial.read();
               unsigned int expectedCount = (hi << 8) | lo;
               unsigned int count = 0;
               if (channel >= 0 && channel < availableDAChannels() && channel < MAX_DA_CHANNELS_
                   && expectedCount <= DA_SEQUENCELENGTH) {
                 while (count < expectedCount && waitForSerial(timeOut_)) {
                   byte msb = Serial.read();
                   if (!waitForSerial(timeOut_)) break;
                   byte lsb = Serial.read();
                   daSequence_[channel][count] = (((uint16_t)(msb & 0x0F)) << 8) | (uint16_t) lsb;
                   count++;
                 }
                 daSequenceLength_[channel] = count;
               }
               Serial.write(byte(38));
               Serial.write(byte(channel));
               Serial.write(highByte(count));
               Serial.write(lowByte(count));
             }
           }
         }
         break;

       // Starts DA (analog) triggered sequence output
       case 39:
         daSequenceNr_ = 0;
         daTriggerNr_ = 0;
         daTriggerState_ = digitalRead(inPin_) == HIGH;
         daTriggerMode_ = true;
         Serial.write(byte(39));
         break;

       case 40:
         Serial.write( byte(40));
         Serial.write( PINC);
         break;
         
       case 41:
         if (waitForSerial(timeOut_)) {
           int pin = Serial.read();  
           if (pin >= 0 && pin <=5) {
              int val = analogRead(pin);
              Serial.write( byte(41));
              Serial.write( pin);
              Serial.write( highByte(val));
              Serial.write( lowByte(val));
           }
         }
         break;
         
       case 42:
         if (waitForSerial(timeOut_)) {
           int pin = Serial.read();
           if (waitForSerial(timeOut_)) {
             int state = Serial.read();
             Serial.write( byte(42));
             Serial.write( pin);
             if (state == 0) {
                digitalWrite(14+pin, LOW);
                Serial.write( byte(0));
             }
             if (state == 1) {
                digitalWrite(14+pin, HIGH);
                Serial.write( byte(1));
             }
           }
         }
         break;

       // Stops DA (analog) triggered sequence output
       case 43:
         daTriggerMode_ = false;
         Serial.write(byte(43));
         Serial.write(daTriggerNr_);
         break;

       }
    }
    
    // In trigger mode, we will blank even if blanking is not on..
    if (triggerMode_) {
      boolean tmp = PIND & inPinBit_;
      if (tmp != triggerState_) {
        if (blankOnHigh_ && tmp ) {
          PORTB = 0;
        }
        else if (!blankOnHigh_ && !tmp ) {
          PORTB = 0;
        }
        else { 
          if (triggerNr_ >=0) {
            PORTB = triggerPattern_[sequenceNr_];
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
        if (! (PIND & inPinBit_))
          PORTB = currentPattern_;
        else
          PORTB = 0;
      }  else {
        if (! (PIND & inPinBit_))
          PORTB = 0;
        else
          PORTB = currentPattern_;
      }
    }

    if (daTriggerMode_) {
      boolean tmp = PIND & inPinBit_;
      if (tmp != daTriggerState_) {
        for (uint8_t ch = 0; ch < availableDAChannels() && ch < MAX_DA_CHANNELS_; ch++) {
          if (daSequenceLength_[ch] > 0) {
            uint16_t code = daSequence_[ch][daSequenceNr_ % daSequenceLength_[ch]];
            analogueOut(ch, (byte)(code >> 8), (byte)(code & 0xFF));
          }
        }
        daSequenceNr_++;
        daTriggerNr_++;
        daTriggerState_ = tmp;
      }
    }
}

 
bool waitForSerial(unsigned long timeOut)
{
    unsigned long startTime = millis();
    while (Serial.available() == 0 && (millis() - startTime < timeOut) ) {}
    if (Serial.available() > 0)
       return true;
    return false;
 }

#if defined TLV5618
// Sets analogue output in the TLV5618
// channel is either 0 ('A') or 1 ('B')
// value should be between 0 and 4095 (12 bit max)
// pins should be connected as described above
void analogueOut(int channel, byte msb, byte lsb) 
{
  digitalWrite(latchPin, LOW);
  msb &= B00001111;
  if (channel == 0)
     msb |= B10000000;
  // Note that in all other cases, the data will be written to DAC B and BUFFER
  shiftOut(dataPin, clockPin, MSBFIRST, msb);
  shiftOut(dataPin, clockPin, MSBFIRST, lsb);
  // The TLV5618 needs one more toggle of the clockPin:
  digitalWrite(clockPin, HIGH);
  digitalWrite(clockPin, LOW);
  digitalWrite(latchPin, HIGH);
}

#elif defined TLV56x8
// Sets analogue output in the TLV5618
// channel is either 0 ('A') or 1 ('B')
// value should be between 0 and 4095 (12 bit max)
// pins should be connected as described above
void analogueOut(int channel, byte msb, byte lsb) 
{
   // Select DAC

    // Configure Channel
    msb &= B00001111;
    if (channel == 0){
        digitalWrite(CS1, LOW);  // Activate DAC 1
        digitalWrite(CS2, HIGH); // 
        msb |= B10000000; // 
        }
    else if (channel == 1){
        digitalWrite(CS1, LOW);  // Activate DAC 2
        digitalWrite(CS2, HIGH); // 
        msb &= B00001111; // 
        }
    // Alternative Channel Selection Logic
    else if (channel == 2){
        digitalWrite(CS1, HIGH);  // Activate DAC 3
        digitalWrite(CS2, LOW); // 
        msb |= B10000000; // 
        }
    else if (channel == 3){
        digitalWrite(CS1, HIGH);  // Activate DAC 4
        digitalWrite(CS2, LOW); //
        msb &= B00001111; // 
    }
    // send data
    shiftOut(dataPin, clockPin, MSBFIRST, msb);
    shiftOut(dataPin, clockPin, MSBFIRST, lsb);

    // End Transmission
    digitalWrite(clockPin, HIGH);
    digitalWrite(clockPin, LOW);

    // Deactivate all DACs
    digitalWrite(CS1, HIGH);
    digitalWrite(CS2, HIGH);
}

#elif defined MCP4728

void analogueOut(int channel, byte msb, byte lsb) {
  if (!mcp_ok) return;
  int ch = channel;
  if (ch < 0 || ch > 3) return;
  uint16_t value12 = ((uint16_t)(msb & 0x0F) << 8) | (uint16_t)lsb;
  MCP4728_channel_t mcp_ch = MCP4728_CHANNEL_A;
  if (ch == 1) mcp_ch = MCP4728_CHANNEL_B;
  else if (ch == 2) mcp_ch = MCP4728_CHANNEL_C;
  else if (ch == 3) mcp_ch = MCP4728_CHANNEL_D;
  mcp.setChannelValue(mcp_ch, value12, MCP4728_VREF_VDD, MCP4728_GAIN_1X,
                      MCP4728_PD_MODE_NORMAL, false);
}

#else

void analogueOut(int channel, byte msb, byte lsb) {}; // noop

#endif

// Reports the fixed voltage range and resolution (DA_MIN_VOLTAGE_MICROV /
// DA_MAX_VOLTAGE_MICROV / DA_NUM_STEPS) for the given channel, if a DA chip
// is compiled in and the channel is valid; otherwise reports 0/0/0 (unknown).
bool getDaVoltageRangeMicroV(int channel, int32_t &minMicroV, int32_t &maxMicroV, uint32_t &numSteps) {
#if defined TLV5618 || defined TLV56x8 || defined MCP4728
  if (channel < 0 || channel >= availableDAChannels()) { minMicroV = 0; maxMicroV = 0; numSteps = 0; return false; }
  minMicroV = DA_MIN_VOLTAGE_MICROV;
  maxMicroV = DA_MAX_VOLTAGE_MICROV;
  numSteps = DA_NUM_STEPS;
  return true;
#else
  minMicroV = 0; maxMicroV = 0; numSteps = 0;
  return false;
#endif
}





/* 
 // This function is called through an interrupt   
void triggerMode() 
{
  if (triggerNr_ >=0) {
    PORTB = triggerPattern_[sequenceNr_];
    sequenceNr_++;
    if (sequenceNr_ >= patternLength_)
      sequenceNr_ = 0;
  }
  triggerNr_++;
}


void blankNormal() 
{
    if (DDRD & B00000100) {
      PORTB = currentPattern_;
    } else
      PORTB = 0;
}

void blankInverted()
{
   if (DDRD & B00000100) {
     PORTB = 0;
   } else {     
     PORTB = currentPattern_;  
   }
}   

*/
  


