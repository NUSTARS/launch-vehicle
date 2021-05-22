function KE = KEcalc(mass,v)
%KECALC
%  Calculates the expected KE at landing in ft-lb because that is what NASA
%  wants for NSL.
%
%  Inputs:
%      mass: Mass of the object (section of rocket) in kg
%      v: Ground hit velocity in m/s. This is either found from OpenRocket,
%      or assumed to be the terminal velocity of the rocket under the main
%      parachute
%  Outputs:
%      KE: Kinetic energy at landing in units of ft-lb

if nargin < 2
    mass = input('Enter mass (kg): ');
    v = input('Enter ground hit velocity (m/s): ');
end

KE = (0.5 * mass * v^2)/1.356;


