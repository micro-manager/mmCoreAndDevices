const long BAUD = 115200;
const int BUFFER_SIZE = 64; // Bytes
const char LINE_TERMINATOR = '\n';

int bufferIndex = 0;
char inputBuffer[BUFFER_SIZE];
char input;

void setup() {
  Serial.begin(BAUD);
  while (!Serial) {
    ; // Wait for serial port to connect.
  }

  Serial.println(F("Echo server ready."));
  Serial.println(F("Enter commands terminated by newline (\\n)"));
}

void loop() {
  if (Serial.available() > 0) {
    input = Serial.read();

    if (input != '\n' && bufferIndex < BUFFER_SIZE - 1) {
      inputBuffer[bufferIndex] = input;
      bufferIndex++;
    } else {
      // Null-terminate the string
      inputBuffer[bufferIndex] = '\0';
      
      // Echo the received command back terminated by \r\n
      Serial.println(inputBuffer);
      clearBuffer();
      
      bufferIndex = 0;
    }
  }
}

void clearBuffer() {
  // Small delay to allow the buffer to fill with any pending characters
  delay(10);
  
  while (Serial.available() > 0) {
    Serial.read();
  }
}
