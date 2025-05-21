% DGAA MOSFET Threshold Voltage Dataset Generator
% Based on Kumar et al. IEEE Trans. Nanotech. 2017 paper

clear all;
close all;
clc;

rng(42); % For reproducibility

% Constants
q = 1.6e-19; % Electron charge (C)
eps0 = 8.854e-14; % Permittivity of free space (F/cm)
eps_si = 11.8 * eps0; % Silicon permittivity
eps_ox = 3.97 * eps0; % Oxide permittivity
hbar = 1.054e-34; % Reduced Planck's constant (J·s)
m0 = 9.1e-31; % Electron rest mass (kg)
kT = 0.0259 * q; % Thermal energy at 300K (J)
Eg = 1.12 * q; % Silicon bandgap (J)
ni = 1.5e10; % Intrinsic carrier concentration (cm^-3)
phi_m = 4.7; % Gate work function (eV)
X_si = 4.05; % Silicon electron affinity (eV)

% Effective masses
ml = 0.97 * m0; % Longitudinal effective mass
mt = 0.19 * m0; % Transverse effective mass
mc = 2*ml*mt/(ml+mt); % Cylindrical mass
mz = ml; % Transport mass

% Number of samples
num_samples = 5000;

% Initialize parameter ranges (realistic values based on paper)
L_min = 10e-7;   % Minimum channel length (10 nm in cm)
L_max = 40e-7;  % Maximum channel length (40 nm in cm)

t_si_min = 5e-7; % Minimum channel thickness (5 nm in cm)
t_si_max = 6e-7;% Maximum channel thickness (6 nm in cm)

t_ox_min = 1e-7; % Minimum oxide thickness (1 nm in cm)
t_ox_max = 3e-7; % Maximum oxide thickness (3 nm in cm)

t_c_min = 2e-7;  % Minimum inner gate radius (2 nm in cm)
t_c_max = 3e-7;  % Maximum inner gate radius (3 nm in cm)

Na_min = 1e15;   % Minimum channel doping (cm^-3)
Na_max = 1e17;   % Maximum channel doping (cm^-3)

Nd = 1e20;       % Fixed source/drain doping (cm^-3)

Vds_min = 0.4;   % Minimum drain voltage (V)
Vds_max = 0.9;   % Maximum drain voltage (V)

% Generate random parameters within ranges
L = L_min + (L_max - L_min) * rand(num_samples, 1); % Channel length
t_si = t_si_min + (t_si_max - t_si_min) * rand(num_samples, 1); % Channel thickness
t_ox = t_ox_min + (t_ox_max - t_ox_min) * rand(num_samples, 1); % Oxide thickness
t_c = t_c_min + (t_c_max - t_c_min) * rand(num_samples, 1); % Inner gate radius
Na = Na_min + (Na_max - Na_min) * rand(num_samples, 1); % Channel doping
Vds = Vds_min + (Vds_max - Vds_min) * rand(num_samples, 1); % Drain voltage

% Calculate derived parameters
Vbi = kT/q * log(Na * Nd / ni^2); % Built-in potential
Vfb = phi_m - (X_si + Eg/(2*q) + kT/q * log(Na/ni)); % Flat-band voltage

% Initialize threshold voltage array
Vth = zeros(num_samples, 1);

% Calculate threshold voltage for each sample
for i = 1:num_samples
    % Calculate alpha1 parameter
    alpha1 = (2 * eps_ox) / (eps_si * t_ox(i) * t_si(i)) * (3 - t_c(i)/(t_si(i) + t_c(i) + 2*t_ox(i)));
    numerator = 1 - (eps_ox/(eps_si * t_si(i) * t_ox(i))) * ...
        ((t_c(i) + t_ox(i))*(2*t_c(i) + t_ox(i)) - ...
        (t_si(i) + 2*t_c(i) + 2*t_ox(i)) * (t_c(i) + t_si(i)/2 + t_ox(i)) + ...
        (t_c(i) + t_si(i)/2 + t_ox(i))^2);
    a1 = numerator / sinh(sqrt(alpha1)*L(i));
    term = (eps_ox/(eps_si * t_si(i) * t_ox(i))) * (3 - t_c(i)/(t_si(i) + t_c(i) + 2*t_ox(i)));
    a2 = 2 * (term / alpha1)^2 * (cosh(sqrt(alpha1)*L(i)) - 1);
    a3 = 1 - (eps_ox/(eps_si * t_si(i) * t_ox(i))) * ...
        ((t_c(i) + t_ox(i))*(2*t_c(i) + t_ox(i)) - ...
        (t_si(i) + 2*t_c(i) + 2*t_ox(i)) * (t_c(i) + t_si(i)/2 + t_ox(i)) + ...
        (t_c(i) + t_si(i)/2 + t_ox(i))^2);
    a4 = 1 + a3 * (term / alpha1 - 1);
    a5 = term;
    a6 = -(a5 * 4 * q * Na(i) / (eps_si * alpha1^2) + ...
        2 * (2*Vbi(i) + Vds(i)) * a5 / alpha1) * ...
        (cosh(sqrt(alpha1)*L(i)) - 1);
    
    % Calculate k0,1*r_c (first zero of Bessel function)
    % For l=0, n=1, the first zero is approximately 2.4048
    k0_1_rc = 2.4048;
    rc = t_c(i) + t_si(i)/2;
    E_l_n_j = Eg/2 + (hbar^2)/(2*rc^2) * (1/mc + 1/mt) * (k0_1_rc)^2;
    Qth = 1e12 * q;
    a7 = E_l_n_j + kT * log(Qth / (q * sqrt(2*mz*kT/(pi*hbar^2)))) + ...
        q * Na(i) * a3 / (eps_si * alpha1);
    a8 = (2*Vbi(i)^2 + 2*Vbi(i)*Vds(i)) * (cosh(sqrt(alpha1)*L(i)) - 1) - Vds(i)^2 + ...
        (2*Vbi(i) + Vds(i)) * q * Na(i) / (eps_si * alpha1) * (cosh(sqrt(alpha1)*L(i)) - 1);
   
    A = a1 * a2^2 - a4^2;
    B = a1^2 * a6 + 2 * a2 * a7;
    C = a1^2 * a8 - a7^2;
    discriminant = B^2 - 4*A*C;
    if discriminant < 0
        Vth_sol = -B/(2*A);
    else
        Vth_sol = (-B + sqrt(discriminant)) / (2*A);
    end
    Vth(i) = Vfb(i) + Vth_sol;
    if Vth(i) < 0
        Vth(i) = abs(Vth(i)); % Take absolute value if negative
    end
end

% Create dataset table
parameter_names = {'Channel_Length_cm', 'Channel_Thickness_cm', 'Oxide_Thickness_cm', ...
                  'Inner_Gate_Radius_cm', 'Channel_Doping_cm3', 'Drain_Voltage_V', 'Threshold_Voltage_V'};
              
dataset = table(L, t_si, t_ox, t_c, Na, Vds, Vth, 'VariableNames', parameter_names);

% Remove any duplicate rows to avoid redundancy
dataset = unique(dataset, 'rows');

% If we lost samples due to duplicates, add more to reach 5000
while height(dataset) < 5000
    % Generate additional samples
    num_needed = 5000 - height(dataset);
    
    % Generate new random parameters
    new_L = L_min + (L_max - L_min) * rand(num_needed, 1);
    new_t_si = t_si_min + (t_si_max - t_si_min) * rand(num_needed, 1);
    new_t_ox = t_ox_min + (t_ox_max - t_ox_min) * rand(num_needed, 1);
    new_t_c = t_c_min + (t_c_max - t_c_min) * rand(num_needed, 1);
    new_Na = Na_min + (Na_max - Na_min) * rand(num_needed, 1);
    new_Vds = Vds_min + (Vds_max - Vds_min) * rand(num_needed, 1);
    
    % Calculate Vfb and Vbi for new samples
    new_Vbi = kT/q * log(new_Na * Nd / ni^2);
    new_Vfb = phi_m - (X_si + Eg/(2*q) + kT/q * log(new_Na/ni));
    
    new_Vth = zeros(num_needed, 1);
    
    for i = 1:num_needed
        % Calculate alpha1 parameter
        alpha1 = (2 * eps_ox) / (eps_si * new_t_ox(i) * new_t_si(i)) * ...
                 (3 - new_t_c(i)/(new_t_si(i) + new_t_c(i) + 2*new_t_ox(i)));
        
        % Calculate a1 to a8 parameters
        % a1
        numerator = 1 ...
    - (eps_ox/(eps_si * new_t_si(i) * new_t_ox(i))) * ( ...
        (new_t_c(i) + new_t_ox(i)) * (2*new_t_c(i) + new_t_ox(i)) ...
      - (new_t_si(i) + 2*new_t_c(i) + 2*new_t_ox(i)) * (new_t_c(i) + new_t_si(i)/2 + new_t_ox(i)) ...
      + (new_t_c(i) + new_t_si(i)/2 + new_t_ox(i))^2 ...
    );
a1 = numerator / sinh(sqrt(alpha1) * new_L(i));

        
        % a2
        term = (eps_ox/(eps_si * new_t_si(i) * new_t_ox(i))) * (3 - new_t_c(i)/(new_t_si(i) + new_t_c(i) + 2*new_t_ox(i)));
        a2 = 2 * (term / alpha1)^2 * (cosh(sqrt(alpha1)*new_L(i)) - 1);
        
        % a3
       a3 = 1 ...
    - (eps_ox/(eps_si * new_t_si(i) * new_t_ox(i))) * ( ...
        (new_t_c(i) + new_t_ox(i)) * (2*new_t_c(i) + new_t_ox(i)) ...
      - (new_t_si(i) + 2*new_t_c(i) + 2*new_t_ox(i)) * (new_t_c(i) + new_t_si(i)/2 + new_t_ox(i)) ...
      + (new_t_c(i) + new_t_si(i)/2 + new_t_ox(i))^2 ...
    );

        % a4
        a4 = 1 + a3 * (term / alpha1 - 1);
        
        % a5
        a5 = term;
        
        % a6
        a6 = -(a5 * 4 * q * new_Na(i) / (eps_si * alpha1^2) + ...
            2 * (2*new_Vbi(i) + new_Vds(i)) * a5 / alpha1) * ...
            (cosh(sqrt(alpha1)*new_L(i)) - 1);
        
        % k0,1*r_c (first zero of Bessel function)
        k0_1_rc = 2.4048;
        
        % Effective radius to mid-channel
        rc = new_t_c(i) + new_t_si(i)/2;
        
        % a7
        E_l_n_j = Eg/2 + (hbar^2)/(2*rc^2) * (1/mc + 1/mt) * (k0_1_rc)^2;
        
        % Critical charge density
        Qth = 1e12 * q; % 1e12 cm^-2
        
        a7 = E_l_n_j + kT * log(Qth / (q * sqrt(2*mz*kT/(pi*hbar^2)))) + ...
            q * new_Na(i) * a3 / (eps_si * alpha1);
        
        % a8
        a8 = (2*new_Vbi(i)^2 + 2*new_Vbi(i)*new_Vds(i)) * (cosh(sqrt(alpha1)*new_L(i)) - 1) - new_Vds(i)^2 + ...
            (2*new_Vbi(i) + new_Vds(i)) * q * new_Na(i) / (eps_si * alpha1) * (cosh(sqrt(alpha1)*new_L(i)) - 1);
        
        % Coefficients for quadratic equation
        A = a1 * a2^2 - a4^2;
        B = a1^2 * a6 + 2 * a2 * a7;
        C = a1^2 * a8 - a7^2;
        
        % Solve quadratic equation for Vth - Vfb
        discriminant = B^2 - 4*A*C;
        
        if discriminant < 0
            Vth_sol = -B/(2*A);
        else
            Vth_sol = (-B + sqrt(discriminant)) / (2*A);
        end
        
        new_Vth(i) = new_Vfb(i) + Vth_sol;
        
        % Ensure Vth is non-negative
        if new_Vth(i) < 0
            new_Vth(i) = abs(new_Vth(i));
        end
    end
    
    % Create temporary table for new samples
    temp_table = table(new_L, new_t_si, new_t_ox, new_t_c, new_Na, new_Vds, new_Vth, ...
                      'VariableNames', parameter_names);
    
    % Combine with existing dataset and remove duplicates again
    dataset = unique([dataset; temp_table], 'rows');
end

% Trim to exactly 5000 samples if we went over
dataset = dataset(1:5000, :);

% Convert units for better readability
dataset.Channel_Length_nm = dataset.Channel_Length_cm * 1e7;
dataset.Channel_Thickness_nm = dataset.Channel_Thickness_cm * 1e7;
dataset.Oxide_Thickness_nm = dataset.Oxide_Thickness_cm * 1e7;
dataset.Inner_Gate_Radius_nm = dataset.Inner_Gate_Radius_cm * 1e7;

% Remove cm-scale columns
dataset.Channel_Length_cm = [];
dataset.Channel_Thickness_cm = [];
dataset.Oxide_Thickness_cm = [];
dataset.Inner_Gate_Radius_cm = [];

% Reorder columns
dataset = dataset(:, {'Channel_Length_nm', 'Channel_Thickness_nm', ...
                     'Oxide_Thickness_nm', 'Inner_Gate_Radius_nm', ...
                     'Channel_Doping_cm3', 'Drain_Voltage_V', 'Threshold_Voltage_V'});

% Save dataset to CSV file
writetable(dataset, 'DGAA_MOSFET_Threshold_Voltage_Dataset.csv');

% Display summary statistics
disp('Summary statistics of generated dataset:');
summary(dataset)

% Plot some relationships to visualize the data
figure;
scatter(dataset.Channel_Length_nm, dataset.Threshold_Voltage_V, 10, 'filled');
xlabel('Channel Length (nm)');
ylabel('Threshold Voltage (V)');
title('Threshold Voltage vs. Channel Length');
grid on;

figure;
scatter(dataset.Channel_Thickness_nm, dataset.Threshold_Voltage_V, 10, 'filled');
xlabel('Channel Thickness (nm)');
ylabel('Threshold Voltage (V)');
title('Threshold Voltage vs. Channel Thickness');
grid on;

figure;
scatter(dataset.Channel_Doping_cm3, dataset.Threshold_Voltage_V, 10, 'filled');
set(gca, 'XScale', 'log');
xlabel('Channel Doping (cm^{-3})');
ylabel('Threshold Voltage (V)');
title('Threshold Voltage vs. Channel Doping');
grid on;