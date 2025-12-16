# How does a drone move across a 2D network using either the Random Waypoint model or the Privacy-Preserving Random model, and how do velocity, pause time, and dummy locations influence its path, distance traveled, and final trajectory?

function [] = mobility(mobility_model, velocity, pausing_time, num_dummy_locations)
    % Validate input parameters
    if mobility_model ~= 0 && mobility_model ~= 1
        error('Invalid mobility model. Use 0 for RWP and 1 for PPR.');
    end

    if velocity < 1 || velocity > 5
        error('Velocity should be between 1 and 5 m/s.');
    end

    if pausing_time < 0 || pausing_time > 10
        error('Pausing time should be between 0 and 10 seconds.');
    end

    if num_dummy_locations ~= 2 && num_dummy_locations ~= 3
        error('Number of dummy locations should be 2 or 3.');
    end

    % Define network size
    network_size = 1000;

    num_lines = 2; % We will simulate two segments

    % Initialize starting point for the drone at (0, 0)
    start_points = zeros(num_lines, 2);

    % Initialize the figure for visualization
    figure;

    for i = 1:num_lines
        if mobility_model == 0
            % Random Waypoint (RWP)
            start_symbol = '*'; % Start position marked with *
            destination_points = rand(1, 2) * network_size;
            plot(destination_points(1), destination_points(2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
        else
            % Privacy-Preserving Random (PPR)
            destination_points = rand(1, 2) * network_size;
            rectangle_points = [start_points(i, :); destination_points];
            dummy_points = generate_dummy_locations(rectangle_points, num_dummy_locations);
            all_points = [start_points(i, :); dummy_points; destination_points];
            plot(all_points(:, 1), all_points(:, 2), 'go-', 'LineWidth', 2);
            start_symbol = 'o'; % Start position marked with o
        end

        hold on;
        plot(start_points(i, 1), start_points(i, 2), ['r', start_symbol], 'MarkerFaceColor', 'r', 'MarkerSize', 8); % Drone position marked every one second

        % Perform drone movement
        point_pos = start_points(i, :);
        total_distance = 0;

        while true
            destination = destination_points;
            dist = norm(destination - point_pos);
            total_distance = total_distance + dist;

            time_needed = dist / velocity;
            num_steps = ceil(time_needed / 1); % One-second interval

            step_vector = (destination - point_pos) / num_steps;

            for j = 1:num_steps
                point_pos = point_pos + step_vector + 10*randn(1, 2); % Random displacement added
                % Ensure the drone stays within the network size
                point_pos = max(0, point_pos);
                point_pos = min(network_size, point_pos);

                plot(point_pos(1), point_pos(2), ['r', start_symbol], 'MarkerFaceColor', 'r', 'MarkerSize', 8);

                pause(1);

                % Check if the drone is within a threshold distance of the destination
                if norm(point_pos - destination_points) <= 10
                    break; % Drone is near the destination, terminate the flight
                end
            end

            pause(pausing_time); % Pause at the destination

            if point_pos == destination_points
                break; % Choose a new destination
            end
            if norm(point_pos - destination_points) <= 10
                break; % Drone is near the destination, terminate the flight
            end
        end

        % Show the final destination in green color
        plot(destination_points(1), destination_points(2), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
    end

    fprintf('Total flying distance from the base: %f meters\n', total_distance);
    fprintf('Total flying time: %d seconds\n', (num_steps * num_lines) + (pausing_time * (num_lines - 1)));

    xlabel('X-axis');
    ylabel('Y-axis');

    if mobility_model == 0
        title('Random Waypoint (RWP) Mobility Model of Drone');
    else
        title('Privacy-Preserving Random (PPR) Mobility Model of Drone');
    end

    axis equal;
    grid on;
end

function dummy_points = generate_dummy_locations(rectangle_points, num_dummy_locations)
    % Generate dummy locations within a rectangle defined by diagonal points
    p1 = rectangle_points(1, :);
    p2 = rectangle_points(2, :);
    min_x = min(p1(1), p2(1));
    max_x = max(p1(1), p2(1));
    min_y = min(p1(2), p2(2));
    max_y = max(p1(2), p2(2));

    dummy_points = rand(num_dummy_locations, 2);
    dummy_points(:, 1) = min_x + dummy_points(:, 1) * (max_x - min_x);
    dummy_points(:, 2) = min_y + dummy_points(:, 2) * (max_y - min_y);
end