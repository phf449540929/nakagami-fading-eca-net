clc
clear
close all 

% Igor Martins Rocha
% Simulation of nakagami-m fading channel

m = 1;      % Controls the format, the higher the m, the weaker the fading
w = 1;      % Controls the spreading
snr = 20;

% --------- Plotar FDP --------------------------- %
x = 0:0.01:5;

m = 0.5;
f = (2*(m^m)*x.^(2*m-1).*exp((-m/w)*x.^2))/(gamma(m)*(w^m)); % FDP
plot (x,f);
hold on;

m = 1;
f = (2*(m^m)*x.^(2*m-1).*exp((-m/w)*x.^2))/(gamma(m)*(w^m)); % FDP
plot (x,f);
hold on;

m = 2;
f = (2*(m^m)*x.^(2*m-1).*exp((-m/w)*x.^2))/(gamma(m)*(w^m)); % FDP
plot (x,f);
hold on;

legend({'m = 0.5', 'm = 1', 'm = 2'});
xlabel('envelope');
ylabel('PDF');
axis([0 3 0 1.5]);
% title('Probability Density Function');