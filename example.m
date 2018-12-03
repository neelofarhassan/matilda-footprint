% load('E:\InstanceSpace_WorkforceScheduling\rawdata.mat')
%load('E:\InstanceSpace_GraphColoring\rawdata.mat');
%
function example(user_directory)
 disp(user_directory);
% disp(feature_file);
% disp(configuration_params_file);
% disp(user_name);
opts.perf.MaxMin = false;               % True if Y is a performance measure, False if it is a cost measure.
opts.perf.AbsPerf = false;              % True if an absolute performance measure, False if a relative performance measure
opts.perf.epsilon = 0.1;                % Threshold of good performance

opts.general.betaThreshold = 0.5;       % Beta-easy threshold

opts.bound.flag = false; % bound outliers within the 5 times of the interquartile range of the data

opts.norm.flag = true; % routine uses a boxcox transformation to normalize the data and z-transformation to scale the data. 

opts.diversity.flag = false;            % Run diversity calculation
opts.diversity.threshold = 0.01;        % Minimum percentage allowed of repeated values [?]

opts.corr.flag = false;
% opts.corr.threshold = 3;              % Top N features (by correlation) per algorithm that are selected
opts.corr.threshold = 5;                % Top N features (by correlation) per algorithm that are selected

opts.clust.flag = false;
opts.clust.KDEFAULT = 10;               % Default maximum number of clusters
% opts.clust.SILTHRESHOLD = 0.55;       % Minimum accepted value for the average silhoute value
opts.clust.SILTHRESHOLD = 0.70;         % Minimum accepted value for the average silhoute value
opts.clust.NTREES = 50;                 % Number of trees for the Random Forest (to determine highest separability in the 2-d projection)
opts.clust.MaxIter = 1000;
opts.clust.Replicates = 100;
opts.clust.UseParallel = false;

opts.footprint.RHO = 10;                % Density threshold
opts.footprint.PI = 0.75;               % Purity threshold
opts.footprint.LOWER_PCTILE = 1;        % Lower distance threshold
opts.footprint.UPPER_PCTILE = 25;       % Higher distance threshold

opts.pbldr.ntries = 10;                 % Number of attempts carried out by PBLDR
opts.pbldr.analytic = false;            % Calculate the analytical or numerical solution
opts.pbldr.cmaopts = bipopcmaes;        % Get the default params for BIPOP-CMA-ES
opts.pbldr.cmaopts.StopFitness = 0;     % Stop if the fitness is 0
opts.pbldr.cmaopts.MaxRestartFunEvals = 0;  % Allow multiple restarts 
opts.pbldr.cmaopts.MaxFunEvals  = 1e4;      % Maximum number of evaluations
opts.pbldr.cmaopts.EvalParallel = 'no';     % This should be kept as 'no'
opts.pbldr.cmaopts.DispFinal = 'off';       % To make BIPOP-CMA-ES silent
disp(opts.pbldr.cmaopts.MaxFunEvals); % Number of function evaluation that the optimization route has available
% out = matilda(X, Y, Ybin, opts);
% out = matilda(X(1:500,:), Y(1:500,:), Ybin(1:500,:), opts);

% feature_file = strcat(num2str(user_directory), '/graph-colouring-features.csv');
% performance_file = strcat(num2str(user_directory, 'graph-colouring-algorithms.csv'));
%disp(feature_file);
%disp(performance_file);
%out = matilda('graph-colouring-features.csv', 'graph-colouring-algorithms.csv', opts);

out = matilda(user_directory, opts);

