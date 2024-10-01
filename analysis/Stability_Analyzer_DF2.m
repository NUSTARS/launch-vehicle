%Stability Calculator (by Victor Yip)

% Constants (all dimensions in meters):
L_n = 0.548;    % Nose Length
F_r = 0.150;    % Fin Root Chord
F_t = 0.033;    % Fin Tip Chord
F_s = 0.12192;  % Fin Semi-Span
S = 0.084;      % Sweep Distance
L_r = 2.65575;  % Rocket Length
L_z = 0.0762;   % Nozzle Length
N = 3;          % # of fins
Cn_n = 0.5;     % Nose Cone Coefficient
t = 0.00635;    % Max Fin Root Thickness
X_tc = 0.00635; % Distance from Fin Leading Edge to Max Thickness
L_red = 0.05995;% Length Reduction @ back
D_noz = 0.08204;% Nozzle Diameter
D_nos = 0.140;  % Nose Base Diameter
D_end = 0.1077; % End Diameter
F_w = 0.00381;    % Fin Width
F_fl = 0.21895; % Fin Front Length

% Calculated Dimensions
F_m = F_s/cos(atan((S+(F_t/2)-(F_r/2))/F_s)); % Fin Mid-Chord Line
L_nf = (L_r-F_r)-L_red; % Nose Tip to Fin Chord Leading Edge

% X (distance from nose tip to component's Cp):
% NOTE: Normal Body and Shoulder Forces are neglected 
X_n = L_n/2;
X_b = L_r - L_red + (L_red/3)*(1+1/(1+D_nos/D_end));
X_fb = L_nf+ (S*(F_r+2*F_t))/(3*(F_r+F_t)) + (1/6)*(F_r+F_t-(F_r*F_t)/(F_r+F_t));

% Component Coefficient of Normal Force (Cn)
Cn_n = 2;
Cn_b = 2*(((D_end/D_nos)^2)-1);
Cn_f = (4*N*(F_s/D_nos)^2)/(1+sqrt(1+(2*F_m/(F_r+F_t))^2));
int = 1+ (D_nos/2)/(F_s+(D_nos/2));
Cn_fb = int*Cn_f;

% Net Coefficient of Normal Force
Cn_net = Cn_n + Cn_b + Cn_fb;
% Net Cp Distance from Nose Tip 
X_net = ((Cn_n*X_n)+(Cn_b*X_b)+(Cn_fb*X_fb))/Cn_net;

% CP POSITION AS FUNC OF ROOT CHORD
% Using F_r as variable (default values for all else):
r_L_nf = @(x)(L_r-x)-L_red;
r_F_m = @(x) F_s/cos(atan((S+(F_t/2)-(x/2))/F_s));
r_Cn_fb = @(x) ((4*N*(F_s/D_nos)^2)/(1+sqrt(1+(2*r_F_m(x)/(x+F_t))^2)))*int;
r_X_fb = @(x) (r_L_nf(x)+ (S*(x+2*F_t))/(3*(x+F_t)) + (1/6)*(x+F_t-(x*F_t)/(x+F_t)));
f_r = @(x)((Cn_n*X_n)+(Cn_b*X_b)+(r_Cn_fb(x)*r_X_fb(x)))/(Cn_b+Cn_n+r_Cn_fb(x));
n_step = 100;
step = 0.3/100;
d = linspace(0,0.5,n_step);
y_r = zeros(1,100);
for i = 1:n_step
    y_r(i) = f_r(d(i));
end
subplot(4,4,1)
plot(d,y_r,'r');
title('Cp Position Relative to Nose Tip');
xlabel('Root Chord Length');
ylabel('Cp Position');

% CP POSITION AS FUNC OF TIP CHORD
t_F_m = @(x) F_s/cos(atan((S+(x/2)-(F_r/2))/F_s));
t_Cn_fb = @(x) ((4*N*(F_s/D_nos)^2)/(1+sqrt(1+(2*t_F_m(x)/(F_r+x))^2)))*int;
t_X_fb  = @(x) (L_nf+ (S*(F_r+2*x))/(3*(F_r+x)) + (1/6)*(F_r+x-(F_r*x)/(F_r+x)));
f_t = @(x) ((Cn_n*X_n)+(Cn_b*X_b)+(t_Cn_fb(x)*t_X_fb(x)))/(Cn_b+Cn_n+t_Cn_fb(x));
y_t = zeros(1,100);
for i = 1:n_step
    y_t(i) = f_t(d(i));
end
subplot(4,4,2)
plot(d,y_t,'b')
title('Cp Position Relative to Nose Tip');
xlabel('Tip Chord Length');
ylabel('Cp Position');

% CP POSITION AS FUNC OF SEMI-SPAN 
se_F_m = @(x) x/cos(atan((S+(F_t/2)-(F_r/2))/x));
se_int = @(x) 1+(D_nos/2)/(x+(D_nos/2));
se_Cn_fb = @(x) ((4*N*(x/D_nos)^2)/(1+sqrt(1+(2*se_F_m(x)/(F_r+F_t))^2)))*se_int(x);
se_X_fb = X_fb;
f_se = @(x) ((Cn_n*X_n)+(Cn_b*X_b)+(se_Cn_fb(x)*se_X_fb))/(Cn_b+Cn_n+se_Cn_fb(x));
y_se = zeros(1,100);
for i = 1:n_step
    y_se(i) = f_se(d(i));
end
subplot(4,4,5)
plot(d,y_se,'g');
title('Cp Position Relative to Nose Tip');
xlabel('Fin Semi-Span');
ylabel('Cp Position');

% CP POSITION AS FUNC OF SWEEP
sw_F_m = @(x) F_s/cos(atan((x+(F_t/2)-(F_r/2))/F_s));
sw_Cn_fb = @(x) int*(4*N*(F_s/D_nos)^2)/(1+sqrt(1+(2*sw_F_m(x)/(F_r+F_t))^2));
sw_X_fb = @(x) (L_nf+ (x*(F_r+2*F_t))/(3*(F_r+F_t)) + (1/6)*(F_r+F_t-(F_r*F_t)/(F_r+F_t)));
f_sw = @(x) ((Cn_n*X_n)+(Cn_b*X_b)+(sw_Cn_fb(x)*sw_X_fb(x)))/(Cn_b+Cn_n+sw_Cn_fb(x));
y_sw = zeros(1,100);
for i = 1:n_step
    y_sw(i) = f_sw(d(i));
end
subplot(4,4,6)
plot(d,y_sw,'c');
title('Cp Position Relative to Nose Tip');
xlabel('Fin Sweep');
ylabel('Cp Position');

%Two-Variable Plots

%ROOT CHORD v.s. TIP CHORD as variables
[R_mesh,T_mesh] = meshgrid(0:step:0.3);
rt_L_nf = (L_r-R_mesh)-L_red;
rt_F_m = F_s./cos(atan((S+(T_mesh./2)-(R_mesh/2))/F_s));
rt_Cn_fb = ((4*N*(F_s/D_nos).^2)./(1+sqrt(1+(2*rt_F_m./(R_mesh+T_mesh)).^2)))*int; 
rt_X_fb = (rt_L_nf+ (S*(R_mesh+2.*T_mesh))./(3*(R_mesh+T_mesh)) + (1/6)*(R_mesh+T_mesh-(R_mesh.*T_mesh)./(R_mesh+T_mesh)));
Y_RT = ((Cn_n*X_n)+(Cn_b*X_b)+(rt_Cn_fb.*rt_X_fb))./(Cn_b+Cn_n+rt_Cn_fb);

subplot(4,4,[3 4 7 8]) % Plotting Surface
surf(R_mesh,T_mesh,Y_RT,'LineStyle','none');
xlabel('Root Chord');
ylabel('Tip Chord');
zlabel('Cp Position');

% SWEEP v.s. TIP CHORD as variables
[SW_mesh,T_mesh] = meshgrid(0:step:0.3);
st_F_m = F_s./cos(atan((SW_mesh+(T_mesh./2)-(F_r/2))/F_s));
st_Cn_fb = ((4*N*(F_s/D_nos).^2)./(1+sqrt(1+(2*st_F_m./(F_r+T_mesh)).^2)))*int;
st_X_fb = (L_nf+ (SW_mesh.*(F_r+2*T_mesh))./(3*(F_r+T_mesh)) + (1/6)*(F_r+T_mesh-(F_r.*T_mesh)./(F_r+T_mesh)));
Y_ST = ((Cn_n*X_n)+(Cn_b*X_b)+(st_Cn_fb.*st_X_fb))./(Cn_b+Cn_n+st_Cn_fb);

subplot(4,4,[9 10 13 14]) % Plotting Surface
surf(SW_mesh,T_mesh,Y_ST,'LineStyle','none');
xlabel('Sweep Distance');
ylabel('Tip Chord');
zlabel('Cp Position');

% SWEEP v.s. ROOT CHORD as variables;
[SW_mesh,R_mesh] = meshgrid(0:step:0.3);
sr_L_nf = (L_r-R_mesh)-L_red;
sr_F_m = F_s./cos(atan((SW_mesh+(F_t/2)-(R_mesh./2))/F_s));
sr_Cn_fb = ((4*N*(F_s/D_nos).^2)./(1+sqrt(1+(2.*sr_F_m./(R_mesh+F_t)).^2)))*int;
sr_X_fb = (sr_L_nf+ (SW_mesh.*(R_mesh+2*F_t))./(3*(R_mesh+F_t)) + (1/6)*(R_mesh+F_t-(R_mesh.*F_t)./(R_mesh+F_t)));
Y_SR = ((Cn_n*X_n)+(Cn_b*X_b)+(sr_Cn_fb.*sr_X_fb))./(Cn_b+Cn_n+sr_Cn_fb);

subplot(4,4,[11 12 15 16]) % Plotting Surface
surf(SW_mesh,R_mesh,Y_SR,'LineStyle','none');
xlabel('Sweep Distance');
ylabel('Root Chord');
zlabel('Cp Position');

