function test4()

    close all

    data = [1.2,  0.7, +1 % nearly colinear
           -0.3,  0.5, -1 % nearly colinear
           -3.0, -1.0, +1
            0.1,  1.0, -1
            3.0,  1.1, -1 % nearly colinear
            2.1, -3.0, +1];

    a = 1; % 0
    b = -2;
    c = -1; % 2000

%   After 100000 iteration: 100%
%
%     a = 2.09; % 60000
%     b = -10.8;
%     c = 5.11; % 99600

%   After 300000 iteration: margin 1
%
%     a = 5.02; % 99600
%     b = -27.6;
%     c = 14.31; % 299600
    
    margin = 0.0; % 1.0
    for it = 1:290000
        
        da = 0;
        db = 0;
        dc = 0;
        
        for idx = 1:6
            
            x = data(idx, 1);
            y = data(idx, 2);
            label = data(idx, 3);

            score = a * x + b * y + c;

            plot_debug(data, a, b, c, x, y)        

            pull = label * (1 - min(label * score, 1));

            da = da + pull * x;
            db = db + pull * y;
            dc = dc + pull * 1;
            
        end

        ders = [da, db, dc] / max(abs([da, db, dc]));
        
        step_size = 0.1; %  / abs(ders(midx));
        a = a + step_size * da;
        b = b + step_size * db;
        c = c + step_size * dc;

        hold on
        plot_debug(data, a, b, c, x, y)
        drawnow
                
        if mod(it, 200) == 0
            accuracy = evaluate_training_accuracy(data, a, b, c);
            fprintf('\nAccuracy at iteration %d: %.2f\n', it, accuracy)
            if accuracy == 1.0; break; end
            da,db,dc
        end    
        
    end
    
    a, b, c, it
    
function perc = evaluate_training_accuracy(data, a, b, c)
    
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
    
    fprintf('%.2f ', scores); fprintf('\n')
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
    