clc; clear; close all;

% ALL UNITS IMPERIAL

% inputs
weight = 35;
apogee = 5000;
drogue_d = 18;
cd = 2.2;
main_alt = 500;
main_d = 120;

% constants
g = 32.17;

% computed values
drogue_A = pi() * drogue_d^2 / 4 / 144;
main_A = pi() * main_d^2 / 4 / 144;
mass = weight / g;

% polyfunc = polyfit([0, t_fill], [pi * drogue_d^2/4 * 1/144, pi * main_d^2/4 * 1/144], 2);
% area_function = @(x) polyval(polyfunc, x);

params = struct("apogee", apogee, ...
                "drogue_A", drogue_A, ...
                "main_alt", main_alt, ...
                "main_A", main_A, ...
                "cd", cd, ...
                "mass", mass);
RecoveryDynamicsFunction(params);

function RecoveryDynamicsFunction(params)
    tspan = [0 1e3]; % Time span for ode45

    % Start at apogee
    x_0 = 0;
    y_0 = params.apogee;
    dx_0 = 0;
    dy_0 = 0;

    initial_conditions = [x_0, y_0, dx_0, dy_0];

    % Solve the ODEs using ode45
   
    options = odeset('Events', @eventFunction, 'RelTol', 1e-6);

    newrhs = @(t,input)odeFunction(t,input,params);

    [time, states] = ode45(newrhs, tspan, initial_conditions, options);

    % Plot results
    plotResults(time, states, params);

    clc;

    fprintf("Terminal Velocity: %.2f ft/s\n", min(states(:,4)))
    fprintf("Total Descent Time: %.2f s\n", time(end))
    fprintf("Main Impact Velocity: %.2f ft/s\n", states(end,4))
end

function rho = density(h)
    if h < 36152
        T = 59 - 0.00356 * h; % F
        p = 2116 * ((T+459.7)/518.6)^5.256; % lbf/ft^2
        rho = p / (1718 * (T + 459.7)); % slug/ft^3
    elseif h < 82345
        T = -70;
        p = 473.1 * exp(1.73-0.000048*h);
        rho = p / (1718 * (T + 459.7)); % slug/ft^3
    else
        rho = -1;
    end
end

function dydt = odeFunction(t, input, params)

    % Extract states
    x = input(1);
    y = input(2);
    vx = input(3);
    vy = input(4);

    rho = density(y);
    fprintf("height %.2f and density %.8f \n", y, rho)

    stage = "";
    if y > params.main_alt % UNDER DROGUE FROM APOGEE – MAIN DEPLOYMENT ALT
        stage = "drogue";
        f_drag = 0.5 * rho * vy^2 * params.cd * params.drogue_A;
    elseif y > 0 % UNDER DROGUE
        stage = "main";
        height_to_open = 100;
        if params.main_alt - y < height_to_open
            computed_area = params.drogue_A + (params.main_A - params.drogue_A) * (params.main_alt - y) / height_to_open;
        else
            computed_area = params.main_A;
        end
        f_drag = 0.5 * rho * vy^2 * params.cd * computed_area;
    elseif y <= 0
        stage = "touchdown";
        dydt = [0; 0; 0; 0];
        return;
    else
        fprintf('fucked up at %d during %s\n', y, stage)
        return;
    end

    f_y = f_drag - params.mass*32.17;
    f_x = 0;

    ax = (1/params.mass) * f_x;
    ay = (1/params.mass) * f_y;

    % ODEs
    dydt = [vx; vy; ax; ay];
end

function [value, isterminal, direction] = eventFunction(~, y)
    % Event function to stop integration when y <= 0
    value = y(2);  % y(2) is the height
    isterminal = 1;  % Halt integration
    direction = -1;  % Negative direction
end

function plotResults(time, states, params)
    % Plot results
    figure('Position', [0, 0, 1000, 1000]);

    % Plotting the first subplot - Height vs. Time
    subplot(2, 1, 1);
    plot(time, states(:, 2), "-")
    ylim([0, params.apogee]);
    xlabel("Time [s]")
    ylabel("Height [ft]")
    hold on;

    % Plot horizontal lines for apogee and main deployment altitude
    yline(params.apogee, '--', 'label', 'Apogee', 'Color', 'b');
    yline(params.main_alt, '--', 'label', 'Main Deployment', 'Color', 'g');
    main_deploy_index = find(states(:, 2) <= params.main_alt, 1, 'first');
    xline(time(main_deploy_index), '--', 'label', '', 'Color', 'g');

    title("Height vs. Time")
    hold off

    % Plotting the second subplot - Velocity vs. Time
    subplot(2, 1, 2);
    min_velocity = min(states(:, 4));
    
    plot(time, states(:, 4), "--r");
    ylim([1.05*min_velocity, 0]);
    xlabel("Time [s]")
    ylabel("Velocity [ft/s]")
    title("Velocity vs. Time")

    hold off
end
