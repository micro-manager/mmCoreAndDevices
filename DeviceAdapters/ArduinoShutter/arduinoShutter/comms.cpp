#include <Arduino.h>

#include "comms.h"

///////////////////////////////////////////////////////////////////////////////////////
/// Serial communications
///////////////////////////////////////////////////////////////////////////////////////
void messageInit(Message& msg) {
  msg.cmd = Command::help;
  msg.isValid = true;
  msg.errorMsg = "";
}

// Read from Serial until untilChar char found or char limit read or timeout reached.
// 
// The function returns true when untilChar is found or the input is length limited.
// Otherwise false is returned. untilChar, if found, is returned as last char in String.
//
// If no char limit is desired, then pass 0 for the `char_limit` argument.
// 
// This function call is non-blocking.
bool readStringUntil(String& input, char untilChar, size_t char_limit) {
  static bool timerRunning;
  static unsigned long timerStart;
  static const unsigned long timeout_ms = 1000; // 1 sec; set to 0 for no timeout

  while (Serial.available()) {
    timerRunning = false;

    char c = Serial.read();
    input += c;
    if (c == untilChar) {
      return true;
    }
    if (char_limit && (input.length() >= char_limit)) {
      return true;
    }
    // Restart timer running if the timeout is non-zero.
    if (timeout_ms > 0) {
      timerRunning = true;
      timerStart = millis();
    }
  }

  if (timerRunning && ((millis() - timerStart) > timeout_ms)) {
    timerRunning = false;
    return true;
  }
  
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Message parsing
///////////////////////////////////////////////////////////////////////////////////////////////////
// Parse the string and convert it to a known message format.
void parseMessage(const String& input, Message& msg) {
  // Terminator must be present because serial input has a char limit;
  // exceeding the limit should produce an invalid command.
  if (input.charAt(input.length() - 1) != LINE_TERMINATOR) {
    msg.isValid = false;
    msg.errorMsg = "No line terminator found";
    return;
  }
  int verbEnd = input.indexOf(' ');
  String verbStr;
  String argStr;
  if (verbEnd == -1) {
    // verbStr is the whole string; get rid of the line terminator.
    verbStr = input.substring(0, input.length() - 1);
  } else {
    verbStr = input.substring(0, verbEnd);
    argStr = input.substring(verbEnd + 1);
  }

  // Parse the verb part of the command
  msg.isValid = true;
  if (verbStr.equalsIgnoreCase("help")) {
    msg.cmd = Command::help;
  } else if (verbStr.equalsIgnoreCase("open")) {
    msg.cmd = Command::open;
  } else if (verbStr.equalsIgnoreCase("close")) {
    msg.cmd = Command::close;
  } else {
    // Handle unrecognized commands
    msg.isValid = false;
    msg.errorMsg = "Unrecognized command: " + input;
    return;
  }
}
