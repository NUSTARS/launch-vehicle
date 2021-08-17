from math import pi
from math import sqrt
import sys

def KEcalc(mass,v):
    '''
    Calculates the expected KE at landing in ft-lbs
    
    Inputs:
        mass: Mass of the object (section of rocket) in kg
        v: Ground hit velocity in m/s. This is either from OpenRocket
           or assumed to be the terminal velocity of the rocket under parachute
    Outputs:
        KE: Kinetic energy at landing
    '''
    KE = (0.5 * mass * v**2)/1.356

    return KE

def terminalV(mass,chuteSize,Cd):
    '''
    Calculates the terminal velocity of a mass under a parachute. Density
    of air is assumed to be 1.225kg/m^3 (STP).

    Only the drag of the main parachute is considered. The drogue
    and separated rocket will also cause some drag which is not considered,
    so this calculation contains an inherant saftey factor. Perhaps you can
    add the diameters of the main and drogue for a more accurate answer?

    Inputs:
        mass: Mass of the object in kg (probably the entire rocket) 
        chuteSize: Diameter of the main parachute in inches
        Cd: Parachute drag coefficient. Default is 2.2 (Fruity Chutes) if left blank
    Outputs:
        v_term: Terminal velocity of the object (m/s)
    '''

    rho = 1.225

    chuteSize = chuteSize/39.37 # convert inches to meters
    A = pi*(chuteSize/2)**2; # Calculate frontal area of parachute
    w = mass*9.81; # Calculate force of gravity
    v_term = sqrt( (2*w)/(rho*A*Cd) ); # Find v when drag equals force of gravity (terminal velocity)

    return v_term

def main():
    print(
        "===================================\n",
        "NUSTARS KINETIC ENERGY CALCULATOR\n"
        "===================================\n"
    )
    print('Which calculation would you like to do?')
    print('1: Use ground hit velocity from OpenRocket')
    print('2: Use terminal velocity as ground hit velocity\n')
    
    op = int(input())

    if op == 1:
        names = []
        KEs = []

        v = float(input('Enter ground hit velocity from OpenRocket'))
        names.append(input('Enter name of section of rocket'))
        mass = float(input('Enter mass of section of rocket in kg'))

    elif op == 2:
        pass
