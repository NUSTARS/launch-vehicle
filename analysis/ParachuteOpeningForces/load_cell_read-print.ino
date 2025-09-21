# include "HX711.h"

// define pins for hx711
#define DT_PIN 35
#define SCK_PIN 34

// initialize other pins
const int kill_pin = 32;
int kill_state = 0;
const int speaker_pin = 0;

HX711 scale;
float sf = -3654.65;   // scaling factor

// set up runtime
unsigned long runTime = 120000;
unsigned long startTime;

void setup() {
  Serial.begin(115200);
  while(!Serial) {
    // wait for serial to open
  }
  Serial.println("Serial Open");

  // Initialize HX711
  scale.begin(DT_PIN, SCK_PIN);
  delay(500);
  scale.set_scale(sf);
  scale.tare(); // zero the scale
  delay(500);
  if (!scale.is_ready()) Serial.println("HX711 Failed");   
  //Serial.println(scale.get_units(), 2); // 2 decimal places

  Serial.println("Done Taring");

  // set up kill button
  pinMode(kill_pin, INPUT_PULLUP);  
  
  // set up PWM for speaker
  analogWriteFrequency(speaker_pin, 440);
  analogWrite(speaker_pin, 0);

  startTime = millis(); // mark start of measurement
}


void loop() {
  unsigned long currentTime = millis();
  kill_state = digitalRead(kill_pin);
  //Serial.println(kill_state);
  delay(200);

  if ((currentTime - startTime) < runTime && kill_state == HIGH) {
    // if under run time read and print
    Serial.println(scale.get_units(), 2); // 2 decimal places
    //delay(200);
  } 
  else if (kill_state == LOW) {
    Serial.println("Kill button pressed");
    analogWrite(speaker_pin, 0);
    while(1) {
    }
  }
  else {  // after runtime go into idle loop
    Serial.println("Timeout");
    analogWrite(speaker_pin, 0);
    while(1) {
    }
  } 
}






