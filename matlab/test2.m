function test2()

    data = [1.2,  0.7, +1 % nearly colinear
           -0.3,  0.5, -1 % nearly colinear
           -3.0, -1.0, +1
            0.1,  1.0, -1
            3.0,  1.1, -1 % nearly colinear
            2.1, -3.0, +1];

    a = 1;
    b = -2;
    c = -1;
    max_iter = 290000;
    
    abcs = zeros(3, max_iter);
    margins = zeros(1, max_iter);
    
    margin = 1.0;
    regularize = 10000000.0; % 1.0
    for it = 1:max_iter
        
        abcs(:, it) = [a; b; c];
        
        idx = randi(6);
        
        x = data(idx, 1);
        y = data(idx, 2);
        label = data(idx, 3);
        
        score = a * x + b * y + c;
        
%         plot_debug(data, a, b, c, x, y)        
        
        pull = 0.0;
        if (label == 1) && (score < margin)
            pull = 1.0; % min(-score, 1.0);
        end
        if (label == -1) && (score > -margin)
            pull = -1.0; % max(-score, -1.0);
        end
        % if pull == 0; continue; end
        
        step_size = 0.01;
        a = a + step_size * (x * pull - a / regularize);
        b = b + step_size * (y * pull - b / regularize);
        c = c + step_size * (1 * pull);

%         hold on
%         plot_debug(data, a, b, c, x, y)
%         drawnow
                
%         if mod(it, 200) == 0
            [accuracy, scores] = evaluate_training_accuracy(data, a, b, c);
%             fprintf('\nAccuracy at iteration %d: %.2f\n', it, accuracy)
%             if accuracy == 1.0; break; end
%         end    

        labels = data(:, 3)';
        margins(it) = max(1 - min(1, scores .* labels));
        
    end
    
    figure, plot(abcs')
    figure, plot(margins)
    fprintf('All done!')
    
function [perc, scores] = evaluate_training_accuracy(data, a, b, c)
    
    scores = zeros(1, 6);
    correct = zeros(1, 6);
    for idx = 1:6
        
        x = data(idx, 1);
        y = data(idx, 2);
        label = data(idx, 3);

        score = a * x + b * y + c;
        scores(idx) = score;
        correct(idx) = sign(score) == label;
        
    end
    
%     fprintf('%.2f ', scores); fprintf('\n')
    perc = sum(correct) / size(data, 1);

function plot_debug(data, a, b, c, x, y)

    m = -a / b;
    q = -c / b;

    plot(data(data(:,3)==+1, 1), data(data(:,3)==+1, 2), 'r*')
    hold on
    plot(data(data(:,3)==-1, 1), data(data(:,3)==-1, 2), 'b*')
    plot(linspace(-4,4), m * linspace(-4,4) + q, 'k.')
    plot(x, y, 'ko')
    axis equal
    xlim([-4, 4])
    ylim([-4, 4])
    hold off
    title(sprintf('a %.4f   b %.4f   c %.4f', a, b, c))
    