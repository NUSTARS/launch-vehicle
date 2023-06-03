% Xiteng Yao 2019, Boston University Rocket Propulsion Group
% Fin flutter speed and related calculations


% function V=finfs(cr,ct,t,b,G,h)
% -- Input Parameters Below --
cr=9.75;     %Root Chord in inches
ct=3.75;     %Tip Chord in inches
t=0.125;     %Thickness in inches
b=4.75;      %Semi-Span in inches
G=380000;    %Shear Modulus in psi
h=3000;      %height in feet



% calculations
S=0.5*(cr+ct)*b;
AR=b^2/S;
Lun=ct/cr;

T=59-0.00356*h;                             % temperature in Farenhert
P=2116/144*(((T+459.7)/518.6)^5.256);       % lbs/ ft 2
a=sqrt(1.4*1716.59*(T+460));                % speed of sound
V=a*sqrt((G/(1.337*AR^3*P*(Lun+1)))/(2*(AR+2)*(t/cr)^3))% fin flutter speed

% end

TEST=1105.26*sqrt((380000/(1.337*0.70*0.70*0.70*13.19*(0.38+1)))/(2*(0.70+2)*(0.125/9.75)*(0.125/9.75)*(0.125/9.75)))