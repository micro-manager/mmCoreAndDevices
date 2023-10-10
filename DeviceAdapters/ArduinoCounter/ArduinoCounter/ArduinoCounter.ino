// Arduino firmware that counts input pulses, relays these to
// the output, but stops doing so after a specified
// amount of pulses has been received.
// Since version 2, also can invert the output.
// Communicates through serial, commands:
//   gnnn - starts pulse counting.  nnn should be an
//          integer number (of any size)
//   s - Stops counting, and reverts to transducing
//       all input to the output
//   i - Info: returns identification string
//   pn - Sets/Reads the flag inverting the input polarity
//          n=i Invert the input to the output
//          n=d Do not invert the input to the output
//          n=? report output polarity



unsigned int version_ = 2;

// pin on which to receive the trigger (2 and 3 can be used with interrupts, although this code does not use interrupts)
const int inPin = 2;


const int outPin = 8;

const unsigned long timeOut = 1000;

unsigned int counter = 0;
unsigned int limit;
boolean counting = false;
boolean inputWas;
boolean invert = false;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  pinMode(inPin, INPUT);
  pinMode(outPin, OUTPUT);

  inputWas = digitalRead(inPin);
}

void loop() {
  while (true) {
    // put your main code here, to run repeatedly:

    if (Serial.available() > 0) {
      int inByte = Serial.read();
      switch (inByte) {

          // go (i.e. start) 'g' followed by max number of TTLs to pass
        case 103:
          if (waitForSerial(timeOut)) {
            limit = Serial.parseInt();
            Serial.write("Starting with ");
            Serial.println(limit, DEC);
            // set limit here
            counting = true;
            counter = 0;
            inputWas = digitalRead(inPin);
            break;
          }

        // stop 's'; i.e. operate in passthrough mode
        case 115:
          counting = false;
          Serial.println("Stopping");
          break;

        // get info 'i'; what version are you?
        case 105:
          Serial.println("ArduinoCounter version 2.0");
          break;

        // Sets/gets output polarity
        case 112:
          if (waitForSerial(timeOut)) {
            int command = Serial.read();
            switch (command) {
              case 105:  // "i"
                invert = true;
                Serial.println("Invert");
                break;

              case 100:  // "d"
                invert = false;
                Serial.println("Direct");
                break;

              case 63:  // "?"
                if (invert) {
                  Serial.println("Invert");
                } else {
                  Serial.println("Direct");
                }
                break;
            }
          }
      }
    }
  }

  if (counting) {
    if (invert) {
      if (inputWas && !(PIND & B00000100)) {
        counter++;
        inputWas = LOW;
        PORTB = 1;
      } else if (!inputWas && (PIND & B00000100)) {        
        counter++;
        inputWas = HIGH;
        if (counter <= limit) {
          PORTB = 0;
        }
      }
    } else { // do not invert output
      if (inputWas && !(PIND & B00000100)) {
        counter++;
        inputWas = LOW;
        if (counter <= limit) {
          PORTB = 0;
        }
      } else if (!inputWas && (PIND & B00000100)) {
        inputWas = HIGH;
        PORTB = 1;
      }
    }
  } else {  // not counting
      if (invert) {
        PORTB = !(PIND & B00000100);
      } else {
        PORTB = (PIND & B00000100) >> 2;
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
