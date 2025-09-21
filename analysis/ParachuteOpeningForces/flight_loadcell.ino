#include "HX711.h"
#include <SD.h>
#include <SPI.h>

// define pins for hx711
#define DT_PIN 35
#define SCK_PIN 34

bool debugSerial = false;

// initialize other pins
const int arm_pin = 22;
int arm_state = 0;
const int speaker_pin = 0;

HX711 scale;
float sf = -3654.65;   // scaling factor

// set up timing
unsigned long runTime = 3600000;  // 1 hour
unsigned long startTime;
//unsigned long lastFlush = 0;
int flushCount = 0;
//const unsigned long flushInterval = 3000;
const int buffer = 5000;

// set up Chip Select
const int chipSelect = BUILTIN_SDCARD;
File data;

///////////////////////////////////////////////////////////////////////////////////
void setup() {
  Serial.begin(115200);
  
  if (debugSerial) {
  while(!Serial) { // wait for serial to open
  }
  Serial.println("Serial Open");
  }

  // set up arm button
  pinMode(arm_pin, INPUT_PULLUP);

  // set up PWM for speaker
  analogWriteFrequency(speaker_pin, 440);

  // check for sd card
  if (!SD.begin(chipSelect)) {
    if (debugSerial) Serial.println("SD init failed");
    while (1) {
      analogWrite(speaker_pin, 128);
      delay(50);
      analogWrite(speaker_pin, 0);
      delay(50);
    }
  }
  
  if (debugSerial) Serial.println("Card initialized");

  // Initialize HX711
  scale.begin(DT_PIN, SCK_PIN);
  scale.set_scale(sf);
  scale.tare(); // zero the scale

  if (debugSerial) Serial.println("Done Taring");


  delay(500);

  // set up arm button
  pinMode(arm_pin, INPUT_PULLUP);

  // set up PWM for speaker
  analogWriteFrequency(speaker_pin, 440);
  //analogWrite(speaker_pin, 128);

  // wait for arming
  while (digitalRead(arm_pin) == HIGH) {
    // wait for arming switch - single beeps
    analogWrite(speaker_pin, 128);
    delay(250);
    analogWrite(speaker_pin, 0);
    delay(1000);
  }

  // Open data file
  data = SD.open("flight_sim.txt", FILE_WRITE);
  if (!data) {
    if (debugSerial) Serial.println("Error opening log");
    while (1) {
    }
  }

  startTime = millis(); // mark start of measurement

  analogWrite(speaker_pin, 128);

}


////////////////////////////////////////////////////////////////////////////////////
void loop() {
  unsigned long currentTime = millis();
  float force = scale.get_units();
  
  arm_state = digitalRead(arm_pin);

// Check time elapsed and button state
  if ((currentTime - startTime) < runTime && arm_state == LOW) {
    // print data to the file
    if (data.print(currentTime) == 0) {
      if (debugSerial) Serial.println("Failed to write time");
    }
    
    if (data.print(",") == 0) {
      if (debugSerial) Serial.println("Failed to write comma");
    }

    if (data.println(force) == 0) {
      if (debugSerial) Serial.println("Failed to write force");
    }
    
    // only flush after an interval
    // if (currentTime - lastFlush >= flushInterval) {
    //   data.flush();
    //   lastFlush = currentTime;
    // }

    if (flushCount >= buffer) {
      data.flush();
      flushCount = 0;
    }
    
    flushCount++;
    delay(10);
  } 

  else if (arm_state == HIGH) {
    data.flush();
    data.close();
    if (debugSerial) Serial.println("Disarmed");
    analogWrite(speaker_pin, 0);
    delay(3000);
    while(1) {
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(250);
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(250);
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(1000);
    }
  }

  else if ((currentTime - startTime) > runTime) {  // after runtime go into idle loop
    data.flush();
    data.close();
    if (debugSerial) Serial.println("Timeout");
    analogWrite(speaker_pin, 0);
    delay(3000);
    while(1) {
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(250);
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(250);
      analogWrite(speaker_pin, 128);
      delay(250);
      analogWrite(speaker_pin, 0);
      delay(1000);
    }
  } 
}



