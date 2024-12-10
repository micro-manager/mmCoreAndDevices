#include <Arduino.h>

class PulseGenerator {
private:
    // Configuration parameters
    uint8_t outputPin = LED_BUILTIN;
    uint8_t triggerPin = 2;
    uint32_t pulseDuration = 5000;     // Pulse duration in microseconds
    uint32_t pulseInterval = 100000;   // Interval between pulses in microseconds
    uint32_t numberOfPulses;  // When set to zero go on until stop
    bool waitForTrigger;        // Flag to wait for input trigger
    bool isRunning;             // Flag to track running state

    // Timer interrupt variables
    IntervalTimer pulseTimer;
    IntervalTimer intervalTimer;

    // State tracking
    volatile bool isPulseActive = false;
    volatile bool isReadyToTrigger = true;
    volatile bool countPulses = false;
    volatile uint32_t pulseNumber = 0;

    // Static callback methods for timer interrupts
    static PulseGenerator* volatile instancePtr;

    // Turn pulse off
    static void stopPulseISR() {
        if (instancePtr) {
            digitalWriteFast(instancePtr->outputPin, LOW);
            instancePtr->isPulseActive = false;
        }
    }

    // Trigger next pulse cycle
    static void intervalISR() {
        if (instancePtr) {
            instancePtr->isReadyToTrigger = true;
        }
    }

    // Trigger pin interrupt handler
    static void triggerPinISR() {
        if (instancePtr && instancePtr->waitForTrigger) {
            instancePtr->isReadyToTrigger = true;
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
        digitalWriteFast(outputPin, LOW);

        if (triggerPin != 255) {
            pinMode(triggerPin, INPUT_PULLDOWN);
            attachInterrupt(digitalPinToInterrupt(triggerPin), triggerPinISR, RISING);
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

        // Simplified configuration confirmation
        Serial.print("Config: ");
        Serial.println(useInputTrigger);
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
        Serial.print("3 ");
        Serial.println(interval);
    }

    void setPulseDuration(uint32_t duration) {
        noInterrupts();
        pulseDuration = duration;
        interrupts();

        // Simplified duration confirmation
        Serial.print("4 ");
        Serial.println(duration);
    }

    void setTriggerMode(bool useInputTrigger) {
        noInterrupts();
        waitForTrigger = useInputTrigger;
        interrupts();

        // Simplified trigger mode confirmation
        Serial.print("5 ");
        Serial.println(useInputTrigger);
    }

    void setNumberOfPulses(uint32_t number) {
      noInterrupts();
      numberOfPulses = number;
      countPulses = number != 0;
      interrupts();

      Serial.print(6);
      Serial.println(number);
    }

    void start() {
        // Reset state
        isReadyToTrigger = true;
        isPulseActive = false;
        isRunning = true;

        // Stop any existing timers
        pulseTimer.end();
        intervalTimer.end();

        // no timers running, so we can set volatile variables without interupting interrupts
        pulseNumber = 0;

        // Start interval timer to manage pulse cycles
        intervalTimer.begin(intervalISR, pulseInterval);

        // Simplified start confirmation
        Serial.println("1");
    }

    void stopNoSerialMessage() {
        // Stop timers
        pulseTimer.end();
        intervalTimer.end();
        
        // Ensure output is low
        digitalWriteFast(outputPin, LOW);
        
        // Reset state
        isRunning = false;
        isPulseActive = false;
        isReadyToTrigger = true;
    }

    void stop() {
        stopNoSerialMessage();

        // Simplified stop confirmation
        Serial.print("2");
        Serial.println(numberOfPulses);
    }

    void run() {
        // Only run if active
        if (!isRunning) return;

        if (countPulses && pulseNumber == numberOfPulses) {
          stopNoSerialMessage();
          return;
        }

        // Check if we're ready to trigger a pulse
        if (isReadyToTrigger) {
            // If waiting for trigger and trigger not received, do nothing
            if (waitForTrigger && digitalReadFast(triggerPin) == LOW) {
                return;
            }

            // Generate pulse
            digitalWriteFast(outputPin, HIGH);
            isPulseActive = true;
            isReadyToTrigger = false;
            //if (countPulses) {
            pulseNumber++;
            //}

            // Set timer to stop pulse after duration
            pulseTimer.begin(stopPulseISR, pulseDuration);
        }
    }

    bool getRunningState() const {
        return isRunning;
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

        switch (commandState) {
            case 0:  // Command byte
                currentCommand = incomingByte;
                switch (incomingByte) {
                    case 1:  // Start
                        pulsegen->start();
                        break;
                    
                    case 2:  // Stop
                        pulsegen->stop();
                        break;
                    
                    case 3:  // Set Interval
                    case 4:  // Set Pulse Duration
                    case 5:  // Set Trigger Mode
                    case 6:  // Number of Triggers, zero for always on
                        commandState = 1;
                        parameterByteCount = 0;
                        parameterBuffer = 0;
                        break;
                    
                    default:
                        break;
                }
                break;

            case 1:  // Receiving 32-bit parameter
                // Construct 32-bit value (little-endian)
                parameterBuffer |= ((uint32_t)incomingByte << (parameterByteCount * 8));
                parameterByteCount++;

                if (parameterByteCount == 4) {
                    switch (currentCommand) {
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
                break;
        }
    }
}

void setup() {
    // Initialize serial for communication
    Serial.begin(115200);
    
    // Give some time for serial to initialize
    delay(1000);

    // Create pulse generator on pin 13 with optional trigger on pin 12
    pulsegen = new PulseGenerator(13, 12);

    // Configure initial defaults
    pulsegen->configure(50000, 500000, 0, false);
}

void loop() {
    // Process any incoming serial commands
    processSerialCommands();

    // Run pulse generator
    pulsegen->run();
}