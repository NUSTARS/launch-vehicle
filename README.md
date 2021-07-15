# NUSTARS
This repository contains useful code for various NUSTARS stuff, mostly launch vehicle related.

## `mkorder.py`
Creates a new blank order form in the current directory.
**Example:**  
Running the following line in a shell (providing mkorder.py is on path) will create a new blank order named
"MM_DD_YY_newOrder.xlsx" in the current directory.  
`mkorder.py newOrder

## `readFlightData.py`
Reads in flight data from the chosen CSV file containing flight data from MissileWorks RRC3 altimeters.

## KE (Needs refractoring )
MATLAB code used to calculate kinetic energy at landing of a falling object. Useful for NSL since NASA requires that
nothing land with a greater KE than 75 ft-lbs. 
