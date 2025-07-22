% This script plots the trajectory for the zermelo navigation problem using GANs 
% Load trajectory data
file_path = 'ZGAN2_test.csv'; %Input generated data
this_traj = readmatrix(file_path);
this_traj = this_traj(7, :);
x = this_traj(2:25);
y = this_traj(27:50);
w1 = this_traj(126:150);
w2 = this_traj(151:175);

% Create the plot
figure;
plot(x, y, 'k-', 'LineWidth', 2); % Improved line style
hold on; grid on;

% Plot markers for start and end points
plot(x(1), y(1), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10); 
text(x(1), y(1), 'A', 'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r', 'FontName', 'Times New Roman');


plot(x(end), y(end), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
text(x(end), y(end), 'B', 'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r', ...
    'FontName', 'Times New Roman');
% Set axis limits, labels, and grid properties
xlim([-2, 2]); 
ylim([-2, 2]);
%axis equal;
pbaspect([1 1 1]); 
xlabel('$r_1$', 'FontName', 'Times New Roman', 'FontSize', 35, ...
    'FontAngle', 'italic', 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('$r_2$', 'FontName', 'Times New Roman', 'FontSize', 35, ...
    'FontAngle', 'italic', 'FontWeight', 'bold', 'Interpreter', 'latex');

% Customize axis properties
ax = gca; 
ax.FontName = 'Times New Roman';
ax.FontSize = 35;
ax.XTick = -2:1:2; 
ax.YTick = -2:1:2;


