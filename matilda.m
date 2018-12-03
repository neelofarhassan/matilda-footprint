%function [out] = matilda(featfile, algofile, opts)
function [out] = matilda(user_directory, opts)
try
diarypath = strcat(user_directory, '/matilda_logs.txt');
diary(diarypath);
% -------------------------------------------------------------------------
% matilda.m
% -------------------------------------------------------------------------
%
% By: Mario Andr�s Mu�oz Acosta
%     School of Mathematics and Statistics
%     The University of Melbourne
%     Australia
%     2018
%
% -------------------------------------------------------------------------
%
% Input parameters:
%
% X:    A cell matrix of features. The first row corresponds to the feature
%       names.
% Y:    A cell matrix of algorithm performance. The first row corresponds
%       to the algorithm names.
% Ybin: A cell matrix with a binary measure of algorithm performance, where
%       '1' is good performance and '0' is bad. The first row corresponds
%       to the algorithm names.
% opts: An structure with all the options.
%
% Output paramters:
% 
% out: An structure with all the outputs
%
% -------------------------------------------------------------------------
featfile = strcat(num2str(user_directory), '/feature.csv');
algofile = strcat(num2str(user_directory), '/performance.csv');
outputfile = strcat(num2str(user_directory), '/coordinates.csv');
startProcess = tic;
% -------------------------------------------------------------------------
%
X = readtable(featfile);
X = X(1:500, :);
Y = readtable(algofile);
Y = Y(1:500, :);
featlabels = X.Properties.VariableNames(2:end);
algolabels = Y.Properties.VariableNames(2:end);

X = sortrows(table2cell(X),1);
Y = sortrows(table2cell(Y),1);
featfiles = X(:,1);
algofiles = Y(:,1);
X = cell2mat(X(:,2:end));
Y = cell2mat(Y(:,2:end));
Y(isnan(Y)) = 1e6;

isvalidY = false(length(algofiles),1);
for i=1:length(featfiles)
    isvalidY = isvalidY | strcmp(featfiles{i},algofiles);
end
algofiles = algofiles(isvalidY);
Y = Y(isvalidY,:);

isvalidX = false(length(featfiles),1);
for i=1:length(algofiles)
    isvalidX = isvalidX | strcmp(algofiles{i},featfiles);
end
X = X(isvalidX,:);
featfiles = featfiles(isvalidX);
% Ybin = cell2mat(Ybin(2:end,:));
% -------------------------------------------------------------------------
%
ninst = size(X,1);
nalgos = size(Y,2);
% -------------------------------------------------------------------------
%
if opts.perf.MaxMin
    [bestPerformace,portfolio] = max(Y,[],2);
    if opts.perf.AbsPerf
        Ybin = Y>=opts.perf.epsilon;
    else
        Ybin = bsxfun(@ge,Y,(1+opts.perf.epsilon).*bestPerformace); % One is good, zero is bad
    end
else
    [bestPerformace,portfolio] = min(Y,[],2);
    if opts.perf.AbsPerf
        Ybin = Y<=opts.perf.epsilon;
    else
        Ybin = bsxfun(@le,Y,(1+opts.perf.epsilon).*bestPerformace); 
    end
end
beta = sum(Ybin,2)>opts.general.betaThreshold*nalgos;
diary OFF;
diary;
% ---------------------------------------------------------------------
% Eliminate extreme outliers, i.e., any point that exceedes 5 times the
% inter quantile range, by bounding them to that value.
disp('-> Removing extreme outliers from the feature values.');
[X, out.bound] = boundOutliers(X, opts.bound);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Normalize the data using Box-Cox and Z-transformations
disp('-> Auto-normalizing the data.');
[X, Y, out.norm] = autoNormalize(X, Y, opts.norm);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Check for diversity, i.e., we want features that have non-repeating
% values for each instance. Eliminate any that have only DIVTHRESHOLD unique values.
[X, out.diversity] = checkDiversity(X, opts.diversity);
featlabels = featlabels(out.diversity.selvars);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Detect correlations between features and algorithms. Keep the top CORTHRESHOLD
% correlated features for each algorithm
[X, out.corr] = checkCorrelation(X, Y, opts.corr);
nfeats = size(X, 2);
featlabels = featlabels(out.corr.selvars);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Detect similar features, by clustering them according to their
% correlation. We assume that the lowest value possible is best, as this 
% will improve the projection into two dimensions. We set a hard limit of
% 10 features. The selection criteria is an average silhouete value above
% 0.65
disp('-> Selecting features based on correlation clustering.');
[X, out.clust] = clusterFeatureSelection(X, Ybin,...
                                         opts.clust);
disp(['-> Keeping ' num2str(size(X, 2)) ' out of ' num2str(nfeats) ' features (clustering).']);
nfeats = size(X, 2);
featlabels = featlabels(out.clust.selvars);
diary OFF;
diary;
% ---------------------------------------------------------------------
% This is the final subset of features. Calculate the two dimensional
% projection using the PBLDR algorithm (Munoz et al. Mach Learn 2018)
disp('-> Finding optimum projection.');
out.pbldr = PBLDR(X, Y, opts.pbldr);
disp('-> Completed - Projection calculated. Matrix A:');
projectionMatrix = cell(3, nfeats+1);
projectionMatrix(1,2:end) = featlabels;
projectionMatrix(2:end,1) = {'Z_{1}','Z_{2}'};
projectionMatrix(2:end,2:end) = num2cell(out.pbldr.A);
disp(' ');
disp(projectionMatrix);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Calculating the algorithm footprints. First step is to transform the
% data to the footprint space, and to calculate the 'space' exafootprint.
% This is also the maximum area possible for a footprint.
disp('-> Calculating the space area and density.');
spaceFootprint = findPureFootprint(out.pbldr.Z, true(ninst,1), opts.footprint);
spaceFootprint = calculateFootprintPerformance(spaceFootprint, out.pbldr.Z, true(ninst,1));
out.footprint.spaceArea = spaceFootprint.area;
out.footprint.spaceDensity = spaceFootprint.density;
disp(['    Area: ' num2str(out.footprint.spaceArea) ' | Density: ' num2str(out.footprint.spaceDensity)]);
% ---------------------------------------------------------------------
% This loop will calculate the footprints for good/bad instances and the
% best algorithm.
diary OFF;
diary;
disp('-> Calculating the algorithm footprints.');
out.footprint.good = cell(1,nalgos);
out.footprint.bad = cell(1,nalgos);
out.footprint.best = cell(1,nalgos);
for i=1:nalgos
    tic;
    out.footprint.good{i} = findPureFootprint(out.pbldr.Z,...
                                              Ybin(:,i),...
                                              opts.footprint);
    out.footprint.bad{i} = findPureFootprint( out.pbldr.Z,...
                                             ~Ybin(:,i),...
                                              opts.footprint);
    out.footprint.best{i} = findPureFootprint(out.pbldr.Z,...
                                              portfolio==i,...
                                              opts.footprint);
    disp(['    -> Algorithm No. ' num2str(i) ' - Elapsed time: ' num2str(toc,'%.2f\n') 's']);
end
diary OFF;
diary;
% ---------------------------------------------------------------------
% Detecting collisions and removing them.
disp('-> Removing collisions.');
for i=1:nalgos
    startBase = tic;
    for j=1:nalgos
        if i~=j
            startTest = tic;
            out.footprint.best{i} = calculateFootprintCollisions(out.footprint.best{i}, ...
                                                                 out.footprint.best{j});
            disp(['    -> Test algorithm No. ' num2str(j) ' - Elapsed time: ' num2str(toc(startTest),'%.2f\n') 's']);
        end
    end
    out.footprint.good{i} = calculateFootprintCollisions(out.footprint.good{i},...
                                                         out.footprint.bad{i});
    out.footprint.bad{i} = calculateFootprintCollisions(out.footprint.bad{i},...
                                                        out.footprint.good{i});
    disp(['-> Base algorithm No. ' num2str(i) ' - Elapsed time: ' num2str(toc(startBase),'%.2f\n') 's']);
end
diary OFF;
diary;
% -------------------------------------------------------------------------
% Calculating performance
disp('-> Calculating the footprint''s area and density.');
performanceTable = zeros(nalgos+2,10);
for i=1:nalgos
    out.footprint.best{i} = calculateFootprintPerformance(out.footprint.best{i},...
                                                   out.pbldr.Z,...
                                                   portfolio==i);
    out.footprint.good{i} = calculateFootprintPerformance(out.footprint.good{i},...
                                                   out.pbldr.Z,...
                                                   Ybin(:,i));
    out.footprint.bad{i} = calculateFootprintPerformance( out.footprint.bad{i},...
                                                   out.pbldr.Z,...
                                                  ~Ybin(:,i));
    performanceTable(i,:) = [calculateFootprintSummary(out.footprint.good{i},...
                                                       out.footprint.spaceArea,...
                                                       out.footprint.spaceDensity), ...
                             calculateFootprintSummary(out.footprint.best{i},...
                                                       out.footprint.spaceArea,...
                                                       out.footprint.spaceDensity)];
end
diary OFF;
diary;
% -------------------------------------------------------------------------
% Beta hard footprints. First step is to calculate them.
disp('-> Calculating beta-footprints.');
out.footprint.easy = findPureFootprint(out.pbldr.Z,  beta, opts.footprint);
out.footprint.hard = findPureFootprint(out.pbldr.Z, ~beta, opts.footprint);
diary OFF;
diary;
% Remove the collisions
out.footprint.easy = calculateFootprintCollisions(out.footprint.easy,...
                                                  out.footprint.hard);
out.footprint.hard = calculateFootprintCollisions(out.footprint.hard,...
                                                  out.footprint.easy);
diary OFF;
diary;
% Calculating performance
disp('-> Calculating the beta-footprint''s area and density.');
out.footprint.easy = calculateFootprintPerformance(out.footprint.easy,...
                                                   out.pbldr.Z,...
                                                   beta);
out.footprint.hard = calculateFootprintPerformance(out.footprint.hard,...
                                                   out.pbldr.Z,...
                                                   ~beta);
performanceTable(end-1,6:10) = calculateFootprintSummary(out.footprint.easy,...
                                                         out.footprint.spaceArea,...
                                                         out.footprint.spaceDensity);
performanceTable(end,6:10) = calculateFootprintSummary(out.footprint.hard,...
                                                       out.footprint.spaceArea,...
                                                       out.footprint.spaceDensity);
out.footprint.performance = cell(nalgos+3,11);
diary OFF;
diary;
out.footprint.performance(1,2:end) = {'Easy_Area',...
                                      'Easy_Area_{norm}',...
                                      'Easy_Density',...
                                      'Easy_Density_{norm}',...
                                      'Easy_Purity',...
                                      'Best_Area',...
                                      'Best_Area_{norm}',...
                                      'Best_Density',...
                                      'Best_Density_{norm}',...
                                      'Best_Purity'};
out.footprint.performance(2:end-2,1) = algolabels;
out.footprint.performance(end-1:end,1) = {'Beta-easy', 'Beta-hard'};
out.footprint.performance(2:end,2:end) = num2cell(performanceTable);
disp('-> Completed - Footprint analysis results:');
disp(' ');
disp(out.footprint.performance);
diary OFF;
diary;
% -------------------------------------------------------------------------
% Algorithm selection. Fit a model that would separate the space into
% classes of good and bad performance. 


% ---------------------------------------------------------------------
% Storing the output data as a csv file
%writetable(array2table(out.pbldr.Z,'VariableNames',{'z_{1}','z_{2}'}),'coordinates.csv');
writetable(array2table(out.pbldr.Z,'VariableNames',{'z1','z2'}),outputfile);
diary OFF;
diary;
% ---------------------------------------------------------------------
% Making all the plots. First, plotting the features and performance as
% scatter plots.
% for i=1:nfeats
%     clf;
%     aux = drawScatter(X(:,i), out.pbldr.Z, featlabels{i});
%     print(gcf,'-dpng',['scatter_' featlabels{i} '.png']);
% end
% 
% for i=1:nalgos
%     clf;
%     aux = drawScatter(Y(:,i), out.pbldr.Z, algolabels{i});
%     print(gcf,'-dpng',['scatter_' algolabels{i} '.png']);
% end
% % Drawing the footprints for good and bad performance acording to the
% % binary measure
% for i=1:nalgos
%     clf;
%     aux = drawGoodBadFootprint(out.footprint.good{i},...
%                          out.footprint.bad{i},...
%                          algolabels{i});
%     print(gcf,'-dpng',['footprint_' algolabels{i} '.png']);
% end
% % Drawing the footprints as portfolio.
% clf;
% aux = drawPortfolioFootprint(out.footprint.best, algolabels);
% print(gcf,'-dpng','footprint_portfolio.png');
% % -------------------------------------------------------------------------
% % 
% disp(['-> Completed! Elapsed time: ' num2str(toc(startProcess)) 's']);
disp('EOF');
catch ME
    disp('EOF')
    rethrow(ME);
    diary OFF;
end
diary OFF;
end

