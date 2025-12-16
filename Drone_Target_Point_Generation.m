# How can we generate target points in a 5×5 grid using a scale-free distribution, then plan the drone’s traversal path using three different strategies—Random (RAND), Nearest-Next-Frontier (NNF), and Density-First (DF)—and visualize how each strategy changes the movement pattern across the clustered subareas?

% -------------------------

%clear;
% network size: width (w) x height (h)
% e.g., 1000 (m) x 1000 (m) by default
w_begin= 0;
w_end = 1000;
h_begin = 0;
h_end = 1000;

% number of cells (subareas): 5-by-5, by default
n_cell = 5;
tot_cell = n_cell * n_cell;
size_cell = w_end / n_cell;

% n number of target points: 
n = 100;

% a set of rectangular subareas: 5-by-5
x_ = linspace(w_begin, w_end - size_cell, n_cell);
ux = [];
for i = 1:n_cell
    ux = [ux, x_]; 
end 
ux = ux';

y_ = ones(1, n_cell);  
uy = [];
for i = 1:n_cell
    uy = [uy, y_ .* (size_cell * (i - 1))];
end 
uy = uy';

% n number of weights: w, n-by-1, uniform
uw = ones(n, 1);

% n number of weights: w, n-by-1, uniform
% -- between the interval (w_begin, w_end) 
w_begin = 0;
w_end = 10;
w = w_begin + (w_end - w_begin) .* rand(n, 1);

% coverage area with radius, r (m), by default 100
r = 100;

% -----------------------
% scale-free distribution 
% -----------------------

% clustering exponent, alpha
alpha = 1.4;

% population, pop  
% -- initialize to zero
pop = ones(tot_cell, 1) - 1;

% probability, prob
% -- initialize to zero
prob = ones(tot_cell, 1) - 1;

% a set of rectangular subareas, 25-by-5
subarea_loc = [ux, uy];

% the first target point is randomly assigned to one of cells
pos_subarea = randi(tot_cell);

pos_x = randi(size_cell) + ux(pos_subarea);
pos_y = randi(size_cell) + uy(pos_subarea);
pop(pos_subarea) = pop(pos_subarea) + 1;

% the first target point - randomly assigned
loc(1, 1) = pos_x;
loc(1, 2) = pos_y;

% generate all scale-free target points (x, y)
for i = 2:n
    % calculate probabilities
    % -- sigma_pop = sum(pop, "all")
    sigma_pop = 0;
    for j = 1: tot_cell
        sigma_pop = sigma_pop + power(pop(j) + 1, alpha);
    end
    for j = 1: tot_cell
        prob(j) = power(pop(j) + 1, alpha) / sigma_pop; %power(sigma_pop, alpha);
        %prob(j) = power(pop(j), alpha) / power(sigma_pop, alpha)
    end
    % sanity check: if total probabilities are one
    %tot_prob = sum(prob, "all")

    % randomly choose one of subareas
    % -- pos_subarea = randi(tot_cell);
    
    % choose one of subareas based on the probability
    % -- generate a random and compare with cumulative probabilities 
    rand_prob = rand(1, 1); % generate between 0 to 1
    cumu_prob = 0; 
    for j = 1: tot_cell
        cumu_prob = cumu_prob + prob(j);
        if (cumu_prob >= rand_prob)
            pos_subarea = j;
            break
        end
    end

    % generate a position within the chosen subarea
    pos_x = randi(size_cell) + ux(pos_subarea);
    pos_y = randi(size_cell) + uy(pos_subarea);
    % increment the population of subarea
    pop(pos_subarea) = pop(pos_subarea) + 1;

    % add a new target point's (x, y) into a row
    loc = [loc; [pos_x, pos_y]];
end    

% draw target points in the first figure
figure;
plot(loc(:, 1), loc(:, 2), "rx")
hold on
set(gca, 'FontSize', 16);
set(gca, 'xTick', [-200:200:1200]);
set(gca, 'yTick', [-200:200:1200]);
xlabel('X', 'FontSize', 16);
ylabel('Y', 'FontSize', 16); 
axis([-200 1200 -200 1200]);
title('Graph 1');

% Number of points
num_points = 100;
hold on
% Generate locations
loc = [rand(num_points, 1)*1000, rand(num_points, 1)*1000];

% Define clusters as a 5x5 grid
clusters = ceil(loc / 200);  % 200 = 1000 / 5
clusters = clusters(:, 1) + (clusters(:, 2)-1)*5;

% Compute the center of each cluster
cluster_centers = zeros(25, 2);
for i = 1:25
    cluster_centers(i, :) = mean(loc(clusters == i, :), 1);
end

% Count the number of points in each cluster (this will be our "density")
cluster_density = histcounts(clusters, 25);

% RAND strategy
rand_order = randperm(25);
% draw the path planning using RAND strategy in the second figure
figure;
plot(cluster_centers(rand_order, 1), cluster_centers(rand_order, 2), '-o');
title('Path planning with RAND strategy');
xlabel('X', 'FontSize', 16);
ylabel('Y', 'FontSize', 16);
axis([-200 1200 -200 1200]);
set(gca, 'FontSize', 16);
set(gca, 'xTick', [-200:200:1200]);
set(gca, 'yTick', [-200:200:1200]);

% NNF strategy
current_point = cluster_centers(1, :);
nnf_order = 1;
for i = 2:25
    remaining_clusters = setdiff(1:25, nnf_order);
    remaining_centers = cluster_centers(remaining_clusters, :);
    distances = sum((remaining_centers - current_point).^2, 2);
    [~, closest_cluster_idx] = min(distances);
    closest_cluster = remaining_clusters(closest_cluster_idx);
    nnf_order = [nnf_order; closest_cluster];
    current_point = cluster_centers(closest_cluster, :);
end
% draw the path planning using NNF strategy in the third figure
figure;
plot(cluster_centers(nnf_order, 1), cluster_centers(nnf_order, 2), '-o');
title('Path planning with NNF strategy');
xlabel('X', 'FontSize', 16);
ylabel('Y', 'FontSize', 16);
axis([-200 1200 -200 1200]);
set(gca, 'FontSize', 16);
set(gca, 'xTick', [-200:200:1200]);
set(gca, 'yTick', [-200:200:1200]);

% DF strategy
[~, df_order] = sort(cluster_density, 'descend');
% draw the path planning using DF strategy in the fourth figure
figure;
plot(cluster_centers(df_order, 1), cluster_centers(df_order, 2), '-o');
title('Path planning with DF strategy');
xlabel('X', 'FontSize', 16);
ylabel('Y', 'FontSize', 16);
axis([-200 1200 -200 1200]);
set(gca, 'FontSize', 16);
set(gca, 'xTick', [-200:200:1200]);
set(gca, 'yTick', [-200:200:1200]);