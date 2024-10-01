function v_term = terminalV(mass,chuteSize,Cd)
%TERMINALV
%  Calculates the terminal velocity of a mass under a parachute. Density
%  of air is assumed to be 1.225kg/m^3 (STP).
%
%  Only the drag of the main parachute is considered. The drogue
%  and separated rocket will also cause some drag which is not considered,
%  so this calculation contains an inherant saftey factor. Perhaps you can
%  add the diameters of the main and drogue for a more accurate answer?
%
%  Inputs:
%      mass: Mass of the object in kg (probably the entire rocket) 
%      chuteSize: Diameter of the main parachute in inches
%      Cd: Parachute drag coefficient. Default is 2.2 (Fruity Chutes) if left blank
%  Outputs:
%      v_term: Terminal velocity of the object (m/s)


if nargin < 3
    Cd = 2.2; % Fruity Chutes drag coefficient
end
if nargin < 2
    mass = input('Enter mass (kg): ');
    chuteSize = input('Enter parachute diameter (in): ');
end

rho = 1.225; % Density of air at standard temperature and pressure

chuteSize = chuteSize/39.37; % Convert parachute diameter from inches to meters
A = pi*(chuteSize/2)^2; % Calculate frontal area of parachute
w = mass*9.81; % Calculate force of gravity
v_term = sqrt( (2*w)/(rho*A*Cd) ); % Find v when drag equals force of gravity (terminal velocity)

