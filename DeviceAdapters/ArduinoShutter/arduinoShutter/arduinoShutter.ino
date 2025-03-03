#include "comms.h"

const int PIN = 2; 
const long BAUD = 115200;
const uint8_t OK    = 0;
const uint8_t ERROR = 1;

// Prints a help message to the screen.
//
// Update this function if LINE_TERMINATOR changes.
void printHelp() {
  Serial.println(F("Available commands:"));
  Serial.println(F("  open\\n"));
  Serial.println(F("  close\\n"));
  Serial.println(F("  help\\n"));
  Serial.println(F(""));
  Serial.println(F("Note: commands must be terminated with a \\n character."));
}

///////////////////////////////////////////////////////////////////////////////////////
/// Main Program
///////////////////////////////////////////////////////////////////////////////////////
String input;
Message msg;

void setup() {
  Serial.begin(BAUD);

  pinMode(PIN, OUTPUT);
  messageInit(msg);

  Serial.println("Shutter ready");
}

void loop() {
    if (readStringUntil(input, LINE_TERMINATOR, CHAR_LIMIT)) {
      parseMessage(input, msg);
      if (msg.isValid) {
        doAction(msg);
        Serial.print(String(OK) + LINE_TERMINATOR);
      } else {
        Serial.println(msg.errorMsg);
        Serial.print(String(ERROR) + LINE_TERMINATOR);
      }

      // Clear the input buffer to prepare for the next command.
      input = "";
    }
}

// Perform the action associated with the message.
void doAction(const Message& msg) {
  switch (msg.cmd) {
    case Command::open:
      digitalWrite(PIN, HIGH);
      break;
    case Command::close:
      digitalWrite(PIN, LOW);
      break;
    case Command::help:
      printHelp();
      break;
    default:
      break;
  }
}
