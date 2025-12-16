# How does a drone choose its traversal path across 25 cluster centers when switching between three different navigation strategies—random order, nearest-next-frontier, and density-first—and how does each strategy change the path drawn and the points scanned in the final plot?

% ... (existing code above)

% Define the selected strategy here (choose one of: 'rand', 'nnf', 'df')
selected_strategy = 'rand'; 

% ... (existing code continues below)

if strcmp(selected_strategy, 'rand')
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
elseif strcmp(selected_strategy, 'nnf')
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
else
    % DF strategy (default if selected_strategy is not recognized)
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
end

% ... (existing code continues below)

% Mark the scan points with '^'
plot(loc(scanned_points, 1), loc(scanned_points, 2), '^', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g');

% ... (continue with the rest of the code to customize the plot)

% Save the final plot
print -depsc sf_topo_final;
savefig('sf_topo_final.fig');