#include <Arduino.h>

static const uint32_t version = 1;

const uint8_t outputPin = LED_BUILTIN;  // Modify as desired.  Teensy LED: LED_BUILTIN;
const bool activateOn = HIGH;             // set to LOW if required by device
const uint8_t inputPin = 2;             // MOdify as desired
const bool inputStateDeviceIsBusy = HIGH;  // set to LOW if required


//  Do not modify below this line
const int activate = activateOn;
const int inActivate = !activateOn;

/**
Note: The pulse pin nr and input pin nr are hard coded.
Change as needed.


Command structure:  first byte indicates the command.
Commands are followed by a uint_32t number (4 bytes)
in big endian format.
The Teensy responds by sending the command nr as a byte,
followed by a 4 byte number (uint32_t in big endian format).
A special command is the number 255, which indicates a request
for the state.  255 is followed by one more byte: the number of 
the command for which the state is requested.  The Teensy 
answers with the command number and the state (as a 4 byte 
uint32_t).


Commands:
0: Version nr.  Returns command nr followed by version number. 
1: Starts running a pulse sequence with features set by other 
    parameters.  Returns command nr.
2: Stops running a pulse sequence.  Returns command nr followed
    by the number of pulses send.
3: Sets the pulse interval in MicroSeconds.  Returns the command nr
    followed by the pulse interval in us.
4. Sets the pulse duration in MicroSeconds.
5. Sets whether or not to wait sending a pulse for the input pin 
    to go high.  Any number > 0 will result in waiting for the input
    pin.
6. Sets the number of pulses to be generated.  When set to zero, 
    pulses will continue until the stop command is received.

*/



class PulseGenerator {
private:
    // Configuration parameters, note that outputPin and triggerPin 
    // will be overwritten in the constructor 
    uint8_t outputPin = LED_BUILTIN;    
    uint8_t triggerPin = 2;
    uint32_t pulseDuration = 1000;     // Pulse duration in microseconds
    uint32_t pulseInterval = 100000;   // Interval between pulses in microseconds
    uint32_t numberOfPulses;  // When set to zero go on until stop
    bool waitForTrigger;        // Flag to wait for input trigger
    bool isRunning;             // Flag to track running state

    // Timer interrupt variables
    IntervalTimer pulseTimer;
    IntervalTimer intervalTimer;

    // State tracking
    volatile bool isReadyToTrigger = true;
    volatile bool countPulses = false;
    volatile bool startOnTrigger = false;
    volatile uint32_t pulseNumber = 0;

    // Static callback methods for timer interrupts
    static PulseGenerator* volatile instancePtr;

    const int inputEdge =  inputStateDeviceIsBusy == LOW ? RISING : FALLING;

    // Turn pulse off
    static void stopPulseISR() {
        if (instancePtr) {
            digitalWriteFast(instancePtr->outputPin, inActivate);
            instancePtr->pulseTimer.end();
        }
    }

    static inline void pulseISR() {
      // Generate pulse
      digitalWriteFast(instancePtr->outputPin, activate);
      instancePtr->isReadyToTrigger = false;
      instancePtr->pulseNumber++;
            
      // Set timer to stop pulse after duration
      instancePtr->pulseTimer.begin(stopPulseISR, instancePtr->pulseDuration);
    }  

    // Trigger next pulse cycle
    static void intervalISR() {
        if (instancePtr) {
            instancePtr->isReadyToTrigger = true;
            // If waiting for trigger and trigger not received, do nothing
            if (instancePtr->waitForTrigger &&  digitalReadFast(instancePtr->triggerPin) == inputStateDeviceIsBusy) {
                return;
            }
            pulseISR();
        }
    }

    // Trigger pin interrupt handler
    static void triggerPinISR() {
      if (instancePtr && instancePtr->startOnTrigger) {
        instancePtr->intervalTimer.begin(instancePtr->intervalISR, instancePtr->pulseInterval);
        instancePtr->startOnTrigger = false;
      }
      if (instancePtr && instancePtr->waitForTrigger && instancePtr->isReadyToTrigger) {
          pulseISR();
      }
    }

public:
    PulseGenerator(uint8_t outPin, uint8_t trigPin = 255) : 
          outputPin(outPin), 
          triggerPin(trigPin), 
          pulseDuration(1000), 
          pulseInterval(10000), 
          waitForTrigger(false), 
          isRunning(false),
          countPulses(false),
          pulseNumber(0) {
        instancePtr = this;
        pinMode(outputPin, OUTPUT);
        digitalWriteFast(outputPin, inActivate);

        if (triggerPin != 255) {
            pinMode(triggerPin, INPUT_PULLDOWN);
            attachInterrupt(digitalPinToInterrupt(triggerPin), triggerPinISR, inputEdge);
        }
    }

    void configure(uint32_t duration, uint32_t interval, uint32_t number, bool useInputTrigger = false) {
        noInterrupts();
        pulseDuration = duration;
        pulseInterval = interval;
        waitForTrigger = useInputTrigger;
        numberOfPulses = number;
        countPulses = number != 0;
        interrupts();
    }

    void setInterval(uint32_t interval) {
        noInterrupts();
        pulseInterval = interval;
        
        // Restart timer if currently running
        if (isRunning) {
            intervalTimer.end();
            intervalTimer.begin(intervalISR, pulseInterval);
        }
        interrupts();

        // Simplified interval confirmation
        Serial.write(3);
        Serial.write((byte *) &interval, 4);
    }

    void reportInterval() {
      noInterrupts();
      uint32_t interval = pulseInterval;  
      interrupts();
      Serial.write(3);
      Serial.write((byte *) &interval, 4);
    }

    void setPulseDuration(uint32_t duration) {
        noInterrupts();
        pulseDuration = duration;
        interrupts();

        // Simplified duration confirmation
        Serial.write(4);
        Serial.write((byte *) &duration, 4);
    }

    void reportPulseDuration() {
      noInterrupts();
      uint32_t duration = pulseDuration;
      interrupts();
      Serial.write(4);
      Serial.write((byte *) &duration, 4);
    }

    void setTriggerMode(bool useInputTrigger) {
        noInterrupts();
        waitForTrigger = useInputTrigger;
        interrupts();

        // Simplified trigger mode confirmation
        Serial.write(5);
        uint32_t tmp = static_cast<uint32_t> (useInputTrigger);
        Serial.write((byte *) &tmp, 4);
    }

    void reportTriggerMode() {
      Serial.write(5);
      noInterrupts();
      bool useInputTrigger = waitForTrigger;
      interrupts();
      uint32_t tmp = static_cast<uint32_t> (useInputTrigger);
      Serial.write((byte *) &tmp, 4);
    }

    void setNumberOfPulses(uint32_t number) {
      noInterrupts();
      numberOfPulses = number;
      countPulses = number != 0;
      interrupts();

      Serial.write(6);
      Serial.write((byte *) &number, 4);
    }

    void reportNumberOfPulses() {
      noInterrupts();
      uint32_t number = numberOfPulses;
      interrupts();
      Serial.write(6);
      Serial.write((byte *) &number, 4);
    }

    void start() {
        // Reset state
        isReadyToTrigger = true;
        isRunning = true;

        // Stop any existing timers
        pulseTimer.end();
        intervalTimer.end();
        // just in case we were within a pulse
        stopPulseISR();

        // no timers running, so we can set volatile variables without interupting interrupts
        noInterrupts();  // Note that triggerPinISR is always running
        pulseNumber = 0;
        interrupts();

        // Start interval timer to manage pulse cycles
        // start first pulse here, otherwise we would wait pulseInterval for the first pulse
        
        if (waitForTrigger && digitalReadFast(triggerPin) == inputStateDeviceIsBusy) {
          startOnTrigger = true;
        } else {
          intervalTimer.begin(intervalISR, pulseInterval);
          pulseISR();
        }

        // Simplified start confirmation
        Serial.write(1);
        // Not really needed, but used for consistent messaging.
        uint32_t tmp = static_cast<uint32_t> (isRunning);
        Serial.write((byte *) &tmp, 4);
    }

    void reportIsRunning() {
      Serial.write(1);
      uint32_t tmp = static_cast<uint32_t> (isRunning);
      Serial.write((byte *) &tmp, 4);
    }

    void reportIsStopped() {
      Serial.write(2);
      uint32_t tmp = static_cast<uint32_t> (!isRunning);
      Serial.write((byte *) &tmp, 4);
    }

    void stopNoSerialMessage() {
        // Stop interval timer
        intervalTimer.end();
        // Do not stop pulseTimer.  If stopped by counter, there surely is 
        // a pulse going, and we do not want to stop that prematurely
        //pulseTimer.end();
        
        // Do not ensure output is low, rely on intervalTimer...
        // digitalWriteFast(outputPin, LOW);
        
        // Reset state
        isRunning = false;
        isReadyToTrigger = false;
    }

    void stop() {
        stopNoSerialMessage();

        // Simplified stop confirmation
        Serial.write(2);
        Serial.write((byte *) &pulseNumber, 4);        
    }

    void run() {
        // Only run if active
        noInterrupts();
        bool active = isRunning;
        bool stop = countPulses && pulseNumber == numberOfPulses;
        interrupts();
        if (!active) 
          return;

        if (stop) {
          stopNoSerialMessage();
          return;
        }
    }

};

// Static instance pointer initialization
PulseGenerator* volatile PulseGenerator::instancePtr = nullptr;

// Global instance
PulseGenerator* pulsegen = nullptr;

// Serial command parsing
void processSerialCommands() {
    static uint8_t commandState = 0;
    static uint8_t currentCommand = 0;
    static uint32_t parameterBuffer = 0;
    static uint8_t parameterByteCount = 0;

    while (Serial.available() > 0) {
      uint8_t incomingByte = Serial.read();

      if (commandState == 0) {
        currentCommand = incomingByte;
        if (currentCommand == 255) // 255 asks about state
          commandState = 2;
        else
          commandState = 1;
        parameterByteCount = 0;
        parameterBuffer = 0;
      } else if (commandState == 1) { // commandState == 1                
        // Receiving 32-bit parameter
        // Construct 32-bit value (little-endian)
        parameterBuffer |= ((uint32_t)incomingByte << (parameterByteCount * 8));
        parameterByteCount++;

        if (parameterByteCount == 4) {
          switch (currentCommand) {
            case 0: 
              Serial.write(0);
              Serial.write((byte *) &version, 4);
              break;

            case 1:  // Start
              pulsegen->start();
              break;
                    
            case 2:  // Stop
              pulsegen->stop();
              break;

            case 3:  // Set Interval
              pulsegen->setInterval(parameterBuffer);
              break;
                        
            case 4:  // Set Pulse Duration
              pulsegen->setPulseDuration(parameterBuffer);
              break;
                        
            case 5:  // Set Trigger Mode
              pulsegen->setTriggerMode(parameterBuffer > 0);
              break;

            case 6:  // Set number of Pulses
              pulsegen->setNumberOfPulses(parameterBuffer);
              break;
          }
                    
          commandState = 0;
        }
      } else if (commandState == 2) {
        switch (incomingByte) {
            case 0: 
              Serial.write(incomingByte);
              Serial.write((byte *) &version, 4);
              break;

            case 1:  // are we running?
              pulsegen->reportIsRunning();
              break;
                    
            case 2:  // are we stopped?
              pulsegen->reportIsStopped();
              break;

            case 3:  // Set Interval
              pulsegen->reportInterval();
              break;
                        
            case 4:  // Set Pulse Duration
              pulsegen->reportPulseDuration();
              break;
                        
            case 5:  // Set Trigger Mode
              pulsegen->reportTriggerMode();
              break;

            case 6:  // Set number of Pulses
              pulsegen->reportNumberOfPulses();
              break;
          }
                    
          commandState = 0;

      }
    }
}

void setup() {
    // Initialize serial for communication
    Serial.begin(115200);
    
    // Give some time for serial to initialize
    delay(1000);

    pinMode(inputPin, INPUT_PULLUP);
    pinMode(outputPin, OUTPUT);

    // Create pulse generator on pin 13 with optional trigger on pin 12
    pulsegen = new PulseGenerator(outputPin, inputPin);

    // Configure initial defaults
    pulsegen->configure(1000, 100000, 0, false);
}

void loop() {
    // Process any incoming serial commands
    processSerialCommands();

    // Run pulse generator
    pulsegen->run();
}