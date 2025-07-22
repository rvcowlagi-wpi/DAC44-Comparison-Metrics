%This script implements the PCA index for the LTI problem
addpath('C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\ECC_2025_conference\PCA_metric');
file_vae = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/svae_test.csv";
file_splitvae = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/splitvae_test.csv";
file_true = "C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\2025_VRNN\case01\set05";
csv_true = dir(fullfile(file_true, 'traj*.csv'));  
real_data = concate_data(file_true,csv_true);
nData = 500;
X = real_data;
X_splitvae = readmatrix(file_splitvae);
X_vae = readmatrix(file_vae);
[Xotd,S,M] = pca_cal(X',3);
[X_vae_gen,S_vae,M_vae] = pca_cal(X_vae,3);
[X_split_gen,S_splitvae,M_splitvae] = pca_cal(X_splitvae,3);
figure;
xLimMin = min([Xotd(:,1); X_vae_gen(:, 1);X_split_gen(:,1)]);
xLimMax = max([Xotd(:,1); X_vae_gen(:, 1);X_split_gen(:,1)]);
yLimMin = min([Xotd(:,2); X_vae_gen(:, 2);X_split_gen(:,2)]);
yLimMax = max([Xotd(:,2); X_vae_gen(:, 2);X_split_gen(:,2)]);
zLimMin = min([Xotd(:,3); X_vae_gen(:, 3);X_split_gen(:,3)]);
zLimMax = max([Xotd(:,3); X_vae_gen(:, 3);X_split_gen(:,3)]);
fontSize_ = 36;
filenames_= {['pca1_JAIS_10dim' num2str(nData) '.png'], ...
	['pca2_JAIS_10dim' num2str(nData) '.png'], ['pca3_JAIS_10dim' num2str(nData) '.png']};
plot3(Xotd(nData+1:end, 1), Xotd(nData+1:end, 2), Xotd(nData+1:end, 3), 'o', 'MarkerSize', 5, 'LineWidth', 1);
hold on;
plot3(Xotd(1:nData, 1), Xotd(1:nData, 2), Xotd(1:nData, 3), 'o', 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 5);
%hold on;
plot3(X_vae_gen(:, 1), X_vae_gen(:, 2), X_vae_gen(:, 3), 'o', 'Color', 'g', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'g');
plot3(X_split_gen(:, 1), X_split_gen(:, 2), X_split_gen(:, 3), 'o', 'Color', 'r', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'r');
xlabel('$\Sigma_{1}$', 'Interpreter', 'latex');
ylabel('$\Sigma_{2}$', 'Interpreter', 'latex');
zlabel('$\Sigma_{3}$', 'Interpreter', 'latex');
legend('Corpus','Training Dataset', 'VAE Generated Data', ...
	'SplitVAE Generated Data','Interpreter', 'latex', 'Location','southeast','FontSize', 40);
make_nice_figures(gcf, gca, fontSize_, [], '$\Sigma_{1}$', '$\Sigma_{2}$',...
	[],[0.05 0.05 0.9*[1 1]],filenames_{1},...
	[xLimMin xLimMax],[yLimMin yLimMax],[zLimMin zLimMax])
hold off;
% Top view
figure;
plot3(Xotd(nData+1:end, 1), Xotd(nData+1:end, 2), Xotd(nData+1:end, 3), 'o', 'MarkerSize', 5, 'LineWidth', 1);
hold on;
plot3(Xotd(1:nData, 1), Xotd(1:nData, 2), Xotd(1:nData, 3), 'o', 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 5);
%hold on;
plot3(X_vae_gen(:, 1), X_vae_gen(:, 2), X_vae_gen(:, 3), 'o', 'Color', 'g', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'g');
plot3(X_split_gen(:, 1), X_split_gen(:, 2), X_split_gen(:, 3), 'o', 'Color', 'r', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'r');
xlabel('$\Sigma_{1}$', 'Interpreter', 'latex');
ylabel('$\Sigma_{2}$', 'Interpreter', 'latex');
zlabel('$\Sigma_{3}$', 'Interpreter', 'latex');
view(0, 90);
legend('Corpus','Training Dataset', 'VAE Generated Data', ...
	'SplitVAE Generated Data','Interpreter', 'latex', 'Location','southeast','FontSize', 40);
xlim([xLimMin xLimMax]);
ylim([yLimMin yLimMax]);
zlim([zLimMin zLimMax]);
make_nice_figures(gcf, gca, fontSize_, [], '$\Sigma_{1}$', '$\Sigma_{2}$',...
	[],[0.05 0.05 0.9*[1 1]],filenames_{2},...
	[xLimMin xLimMax],[yLimMin yLimMax],[zLimMin zLimMax])
hold off;
% Lateral view (side view along z-axis)
figure;
plot3(Xotd(nData+1:end, 1), Xotd(nData+1:end, 2), Xotd(nData+1:end, 3), 'o', 'MarkerSize', 5, 'LineWidth', 1);
hold on;
plot3(Xotd(1:nData, 1), Xotd(1:nData, 2), Xotd(1:nData, 3), 'o', 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', 5);
%hold on;
plot3(X_vae_gen(:, 1), X_vae_gen(:, 2), X_vae_gen(:, 3), 'o', 'Color', 'g', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'g');
plot3(X_split_gen(:, 1), X_split_gen(:, 2), X_split_gen(:, 3), 'o', 'Color', 'r', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'r');
xlabel('$\Sigma_{1}$', 'Interpreter', 'latex');
ylabel('$\Sigma_{2}$', 'Interpreter', 'latex');
zlabel('$\Sigma_{3}$', 'Interpreter', 'latex');
view(90, 0);
legend('Corpus','Training Dataset', 'VAE Generated Data', ...
	'SplitVAE Generated Data','Interpreter', 'latex', 'Location','southeast','FontSize', 40);
make_nice_figures(gcf, gca, fontSize_, [], '$\Sigma_{1}$', '$\Sigma_{2}$',...
	[],[0.05 0.05 0.9*[1 1]],filenames_{3},...
	[xLimMin xLimMax],[yLimMin yLimMax],[zLimMin zLimMax])
hold off;

function concatenated_matrix = concate_data(file_true,csv_files)
num_files = length(csv_files);

if num_files == 0
    error('No CSV files found in the directory.');
end

% Read the first file to determine 'n'
first_file = readmatrix(fullfile(file_true, csv_files(1).name)); 
[rows, n] = size(first_file); 
if rows ~= 10 %Change 10 to 100 for set09
    error('Each CSV file must be 10 × n in size.');
end

% Initialize a matrix of size (4n × 1000)
concatenated_matrix = zeros(10 * n, num_files);%Change 10 to 100 for set09

% Loop through each CSV file
for i = 1:num_files
    data = readmatrix(fullfile(file_true, csv_files(i).name));
    reshaped_data = reshape(data, [], 1);
    concatenated_matrix(:, i) = reshaped_data;
end
end