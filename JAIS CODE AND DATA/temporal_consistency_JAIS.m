% This script implements the NDRR performance index
addpath('C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\ECC_2025_conference\PCA_metric');
file_vae = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/svae_test.csv";
file_splitvae = "C:/Users/nubapat/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2024-JNL-NUBapat/Archives/Scatter-Plots/splitvae_test.csv";
file_true = "C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\2025_VRNN\case01\set05";%set07 set09
file = "C:\Users\nubapat\OneDrive - Worcester Polytechnic Institute (wpi.edu)\2025_VRNN\case01\set05\systemData.mat";%set07 set09
data = load(file);
A = data.A;
G = data.G;
dt = 0.1;            
svae_data = readmatrix(file_vae);
splitvae_data = readmatrix(file_splitvae);
[noise_index,Xs] = compute_dynamics_residual_with_noise(svae_data, A, dt);
fprintf('Noise-aware Dynamics Residual Ratio = %.4f\n', noise_index);
function [noise_ratio,X] = compute_dynamics_residual_with_noise(data_folder, A, dt)
    residual_power = 0;
    signal_power = 0;
    for i = 1:size(data_folder,1)
        X = reshape(data_folder(i,:), 10, 1001); % reshape(data_folder(i,:), 100, 1001) for set09
        T = size(X, 2);
        dXdt = diff(X, 1, 2) / dt;       
        AX = A * X(:, 1:end-1);           
        noise_est = dXdt - AX;             
        residual_power = residual_power + sum(noise_est(:).^2);
        signal_power = signal_power + sum(dXdt(:).^2);
    end
    noise_ratio = residual_power / signal_power;
end
function data = load_all_csvs(folder_path)
    files = dir(fullfile(folder_path, '*.csv'));
    num_files = length(files);
    data = cell(num_files, 1);
    for i = 1:num_files
        filepath = fullfile(folder_path, files(i).name);
        data{i} = readmatrix(filepath); 
    end
end