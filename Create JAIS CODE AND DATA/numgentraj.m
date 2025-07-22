%This file plots the numerically simulated trajectory against the generated
%trajectory
file_path = 'C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/ZVAE_test_500.csv';
R = readmatrix(file_path);
x1nd = zeros(25, 1);
x2nd = zeros(25, 1);
d = zeros(6,1);
dt = zeros(6,1);

for i = 1:1
    k = 1;
    figure;
    current_R = R(i+4, :)';
    [q_sim_traj, X1, Y1, X,y_o] = numgen(current_R);
    arrow_scale = 10;
    x1n = q_sim_traj(:, 1);
    x2n = q_sim_traj(:, 2);
    for j = 1:4:length(x1n)
        x1nd(k) = x1n(j);
        x2nd(k) = x2n(j);
        k = k + 1;
    end
    % Compute the average Euclidean distance between the two trajectories (d1_1)
    d1_1 = mean(sqrt((X1 - x1nd).^2 + (Y1 - x2nd).^2));
    
    % Compute the average magnitude of trajectory A (d1_2)
    d1_2 = mean(sqrt(X1.^2 + Y1.^2));
    
    % Calculate the percent deviation
    d2 = (d1_1 / d1_2) * 100;
    d(i) = d2;
    dt(i) = (abs(y_o(2)-current_R(200))/current_R(200))*100;
    plot(x1nd, x2nd, '-o', 'MarkerSize', 4);  % Customize the plot style as needed
    hold on
    plot(X1, Y1, 'r', 'LineWidth', 1);
    hold on
    quiver(X(:, 1), X(:, 2), arrow_scale * X(:, 3), arrow_scale * X(:, 4), 'k', 'AutoScale', 'off');
    
    % Set the x and y-axis limits for the quiver plot
    xlim([-2, 2]);
    ylim([-2, 2]);
    pbaspect([1 1 1]); 
    xlabel('$r_1$', 'FontName', 'Times New Roman', ...
    'FontSize', 35, 'FontAngle', 'italic', 'FontWeight', 'bold', 'Interpreter', 'latex');
    ylabel('$r_2$', 'FontName', 'Times New Roman', ...
    'FontSize', 35, 'FontAngle', 'italic', 'FontWeight', 'bold', 'Interpreter', 'latex');
    ax = gca; 
    ax.XTick = -2:1:2; 
    ax.YTick = -2:1:2; 
    ax.FontName = 'Times New Roman';
    ax.FontSize = 35;
    lgd = legend('Numerically Simulated Trajectory', 'Generated Trajectory', ...
    'Interpreter', 'latex', 'FontSize', 21);
    grid on;    
     
    text_position_x = -1.2; % X position for the text
    text_position_y = -1.2; % Y position for the text
   text(text_position_x, text_position_y, ...
['$\Delta r = ', num2str(d2, '%.2f'), '\%$, $\Delta t = ', num2str(dt(i), '%.2f'), '\%$'], ...
'Interpreter', 'latex', 'FontSize', 25, 'FontName', 'Times New Roman', 'BackgroundColor', 'white');

   figtitle = 'C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\2024-JNL-NUBapat\Figures\ZVAE_4000_fig_4_new.png';
   exportgraphics(gca, figtitle, 'Resolution', 300);
end


