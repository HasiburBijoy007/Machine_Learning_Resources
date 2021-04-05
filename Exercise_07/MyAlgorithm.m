%MyAlgorithm
%FIRST STEP - CENTROID ASSIGNMENT STEP........
% c(i) := j that minimizes || X(i) - centroids(j) || .^ 2

function Cidx = FindClosestCentroids(X, centroids)
K = size(centroids, 1);                 % 3
Cidx = zeros(size(X,1), 1);              %(m,1)/(300,1)
m = length(Cidx);                        % 300
lengths_to_centroids = zeros(K, 1);     %(3,1)

for i=1:m
  for k=1:K
    lengths_to_centroids(k) = sum((X(i,:) - centroids(k, :)) .^ 2);
  end
  [dummy, ix] = min(lengths_to_centroids);
  Cidx(i) = ix;
end

end


%SECOND STEP - COMPUTE CENTROID MEANS.......
%centroid(k) = (1/Cidx(k)) * sum(X(i) which are remains in that particular Centroid)..

function centroids = computeCentroids(X, Cidx, K)
[m n] = size(X);
centroids = zeros(K, n);

for k=1:K
  %Tot_nums = X(find(idx==k), :);
  %centroids(k, :) = (1/length(Tot_nums)) * sum(Tot_nums);
  centroids(k, :) = mean(X(find(Cidx==k), :));
end

end



%FULL K-MEANS CLUSTERING WITH GRAPH........
function [centroids, idx] = runkMeans(X, initial_centroids, ...
                                      max_iters, plot_progress)
if ~exist('plot_progress', 'var') || isempty(plot_progress)
    plot_progress = false;
end

% Plot the data if we are plotting progress
if plot_progress
    figure;
    hold on;
end

% Initialize values
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);

% Run K-Means
for i=1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    % Optionally, plot progress here
    if plot_progress
        plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
        previous_centroids = centroids;
        fprintf('Press enter to continue.\n');
        pause;
    end
    
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
end

% Hold off if we are plotting progress
if plot_progress
    hold off;
end

end


