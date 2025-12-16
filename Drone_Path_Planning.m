# How can a drone navigate through 25 spatial clusters—formed from randomly distributed points—by following either a random visiting order or a nearest-next-frontier strategy, and how does the chosen method influence the path it generates?

% ... (existing code above)

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

% Selected Strategy
selected_strategy = 'nnf'; % Change this to 'rand' or 'nnf' for different strategies

% Start the drone at the origin (0, 0)
current_drone_position = [0, 0];

if strcmp(selected_strategy, 'rand')
    % RAND strategy
    rand_order = randperm(25);
    drone_path = cluster_centers(rand_order, :);
    strategy_title = 'RAND';
elseif strcmp(selected_strategy, 'nnf')
    % NNF strategy
    visited_clusters = false(25, 1);
    visited_clusters(1) = true;
    drone_path = cluster_centers(1, :);
    
    for i = 2:25
        remaining_clusters = setdiff(1:25, find(visited_clusters));
        remaining_centers = cluster_centers(remaining_clusters, :);
        distances = vecnorm(remaining_centers - current_drone_position, 2, 2);
        
        [~, closest_cluster_idx] = min(distances);
        closest_cluster = remaining_clusters(closest_cluster_idx);
        visited_clusters(closest_cluster) = true;
        
        current_drone_position = cluster_centers(closest_cluster, :);
        drone_path = [drone_path; current_drone_position];
    end
    
    strategy_title = 'NNF';
else
    % Default to NNF strategy if the selected strategy is not recognized
    disp('Invalid strategy selected. Defaulting to NNF strategy.');
    visited_clusters = false(25, 1);
    visited_clusters(1) = true;
    drone_path = cluster_centers(1, :);
    
    for i = 2:25
        remaining_clusters = setdiff(1:25, find(visited_clusters));
        remaining_centers = cluster_centers(remaining_clusters, :);
        distances = vecnorm(remaining_centers - current_drone_position, 2, 2);
        
        [~, closest_cluster_idx] = min(distances);
        closest_cluster = remaining_clusters(closest_cluster_idx);
        visited_clusters(closest_cluster) = true;
        
        current_drone_position = cluster_centers(closest_cluster, :);
        drone_path = [drone_path; current_drone_position];
    end
    
    strategy_title = 'NNF';
end

% Draw the path planning using the selected strategy
figure;
plot(drone_path(:, 1), drone_path(:, 2), '-o');
title(['Path planning with ' strategy_title ' strategy']);
xlabel('X', 'FontSize', 16);
ylabel('Y', 'FontSize', 16);
axis([-200 1200 -200 1200]);
set(gca, 'FontSize', 16);
set(gca, 'xTick', [-200:200:1200]);
set(gca, 'yTick', [-200:200:1200]);

% ... (continue with the rest of the code to customize the plot)

% Save the final plot
print -depsc sf_topo_final;
savefig('sf_topo_final.fig');