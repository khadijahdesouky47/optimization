function newton_raphson_kowalik()
    % Initial setup
    rng(42); % For reproducibility
    a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246];
    b = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16];
    epsilon = 1e-4;
    max_iter = 1000;
    lower_bound = 0;
    upper_bound = 0.42;

    % Initial guesses
    x0 = rand(1, 4) * (upper_bound - lower_bound) + lower_bound;
    x1 = rand(1, 4) * (upper_bound - lower_bound) + lower_bound;
    x2 = rand(1, 4) * (upper_bound - lower_bound) + lower_bound;

    % Methods
    methods = {@newton_raphson, @hestenes_stiefel, @polak_ribiere, @fletcher_reeves, @gradient_descent};
    method_names = {'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Fletcher-Reeves', 'Gradient Descent'};

    % Running and plotting
    results = cell(length(methods), 3);
    for i = 1:length(methods)
        method = methods{i};
        method_name = method_names{i};
        
        fprintf('Running %s method...\n', method_name);
        
        [x_min0, steps0, iter0, time0] = run_method(method, x0, a, b, epsilon, max_iter, lower_bound, upper_bound);
        [x_min1, steps1, iter1, time1] = run_method(method, x1, a, b, epsilon, max_iter, lower_bound, upper_bound);
        [x_min2, steps2, iter2, time2] = run_method(method, x2, a, b, epsilon, max_iter, lower_bound, upper_bound);
        
        % Store results
        results{i, 1} = method_name;
        results{i, 2} = mean([iter0, iter1, iter2]);
        results{i, 3} = mean([time0, time1, time2]);
        
        % Plotting
        plot_steps_in_figure(steps0, [method_name, ' - Initial guess 1']);
        plot_steps_in_figure(steps1, [method_name, ' - Initial guess 2']);
        plot_steps_in_figure(steps2, [method_name, ' - Initial guess 3']);
    end
    
    % Display benchmark table
    disp('Benchmark Results:');
    disp(table(results(:,1), results(:,2), results(:,3), 'VariableNames', {'Method', 'AvgIterations', 'AvgTime'}));
end

function [x_min, steps, iterations, time_taken] = run_method(method, x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    tic;
    [x_min, steps] = method(x0, a, b, epsilon, max_iter, lower_bound, upper_bound);
    time_taken = toc;
    iterations = size(steps, 1) - 1;
end

function [x_min, steps] = newton_raphson(x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    x = x0;
    steps = x;
    for k = 1:max_iter
        [f, grad, hess] = kowalik_function(x, a, b);
        delta_x = -hess \ grad;
        x = x + delta_x;
        x = max(min(x, upper_bound), lower_bound);  % Ensure x stays within bounds
        steps = [steps; x];
        if norm(delta_x) < epsilon
            break;
        end
    end
    x_min = x;
end

function [x_min, steps] = hestenes_stiefel(x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    x = x0;
    steps = x;
    [~, grad_old, hess] = kowalik_function(x, a, b);
    d = -grad_old;
    for k = 1:max_iter
        alpha = - (grad_old' * d) / (d' * hess * d);
        x = x + alpha * d;
        x = max(min(x, upper_bound), lower_bound);  % Ensure x stays within bounds
        steps = [steps; x];
        [~, grad_new, hess] = kowalik_function(x, a, b);
        if norm(grad_new) < epsilon
            break;
        end
        beta = (grad_new' * (grad_new - grad_old)) / (d' * (grad_new - grad_old));
        d = -grad_new + beta * d;
        grad_old = grad_new;
    end
    x_min = x;
end

function [x_min, steps] = polak_ribiere(x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    x = x0;
    steps = x;
    [~, grad_old] = kowalik_function(x, a, b);
    d = -grad_old;
    for k = 1:max_iter
        alpha = line_search(x, d, a, b);
        x = x + alpha * d;
        x = max(min(x, upper_bound), lower_bound);  % Ensure x stays within bounds
        steps = [steps; x];
        [~, grad_new] = kowalik_function(x, a, b);
        if norm(grad_new) < epsilon
            break;
        end
        beta = (grad_new' * (grad_new - grad_old)) / (grad_old' * grad_old);
        d = -grad_new + beta * d;
        grad_old = grad_new;
    end
    x_min = x;
end

function [x_min, steps] = fletcher_reeves(x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    x = x0;
    steps = x;
    [~, grad_old] = kowalik_function(x, a, b);
    d = -grad_old;
    for k = 1:max_iter
        alpha = line_search(x, d, a, b);
        x = x + alpha * d;
        x = max(min(x, upper_bound), lower_bound);  % Ensure x stays within bounds
        steps = [steps; x];
        [~, grad_new] = kowalik_function(x, a, b);
        if norm(grad_new) < epsilon
            break;
        end
        beta = (grad_new' * grad_new) / (grad_old' * grad_old);
        d = -grad_new + beta * d;
        grad_old = grad_new;
    end
    x_min = x;
end

function [x_min, steps] = gradient_descent(x0, a, b, epsilon, max_iter, lower_bound, upper_bound)
    x = x0;
    steps = x;
    alpha = 0.001; % Learning rate
    for k = 1:max_iter
        [~, grad] = kowalik_function(x, a, b);
        x = x - alpha * grad;
        x = max(min(x, upper_bound), lower_bound);  % Ensure x stays within bounds
        steps = [steps; x];
        if norm(grad) < epsilon
            break;
        end
    end
    x_min = x;
end

function alpha = line_search(x, d, a, b)
    % Line search to find optimal alpha
    alpha = 1;
    c1 = 1e-4;
    c2 = 0.9;
    max_iter = 100;
    for i = 1:max_iter
        [f, grad] = kowalik_function(x + alpha * d, a, b);
        if f <= kowalik_function(x, a, b) + c1 * alpha * grad' * d && ...
           grad' * d >= c2 * grad' * d
            break;
        end
        alpha = alpha / 2;
    end
end

function [f, grad, hess] = kowalik_function(x, a, b)
    f = 0;
    grad = zeros(4,1);
    hess = zeros(4,4);
    for i = 1:length(a)
        d = 1 + b(i) * x(3) + x(4) * b(i)^2;
        e = 1 + b(i) * x(2);
        r = a(i) - x(1) * e / d;
        f = f + r^2;
        grad(1) = grad(1) - 2 * r * e / d;
        grad(2) = grad(2) + 2 * r * x(1) * b(i) / d;
        grad(3) = grad(3) - 2 * r * x(1) * b(i) / d;
        grad(4) = grad(4) + 2 * r * x(1) * e * b(i)^2 / d^2;
        
        hess(1,1) = hess(1,1) + 2 * (e / d)^2;
        hess(1,2) = hess(1,2) - 2 * b(i) * e / d^2;
        hess(1,3) = hess(1,3) + 2 * b(i) * e / d^2;
        hess(1,4) = hess(1,4) - 2 * e^2 * b(i)^2 / d^3;
        
        hess(2,2) = hess(2,2) + 2 * (x(1) * b(i) / d)^2;
        hess(2,3) = hess(2,3) - 2 * (x(1) * b(i) / d)^2;
        hess(2,4) = hess(2,4) + 2 * x(1)^2 * b(i)^2 / d^3;
        
        hess(3,3) = hess(3,3) + 2 * (x(1) * b(i) / d)^2;
        hess(3,4) = hess(3,4) - 2 * x(1)^2 * b(i)^2 / d^3;
        
        hess(4,4) = hess(4,4) + 2 * x(1)^2 * e^2 * b(i)^2 / d^4;
    end
    hess(2,1) = hess(1,2);
    hess(3,1) = hess(1,3);
    hess(4,1) = hess(1,4);
    hess(3,2) = hess(2,3);
    hess(4,2) = hess(2,4);
    hess(4,3) = hess(3,4);
end

function plot_steps_in_figure(steps, display_name)
    figure;
    hold on;
    scatter3(steps(:,1), steps(:,2), steps(:,3), 36, steps(:,4), 'filled');
    for i = 1:size(steps, 1)-1
        plot3(steps(i:i+1,1), steps(i:i+1,2), steps(i:i+1,3), 'Color', 'b');
    end
    colorbar;
    xlabel('x1');
    ylabel('x2');
    zlabel('x3');
    title([display_name, ' Steps for Minimizing Kowalik Function']);
    grid on;
    hold off;
end
