function [mean_AP, top_1, top_5] = single_evaluation(first_dist_file)

%***********************************************%
% This code runs on the VeRi-776 dataset. 
% This code uses post fuse the results of CNN + CN + SIFT.
% We use the mAP and hit-1/5 rate as evaluation
%***********************************************%
%% load query hist and test hist
nQuery = 1678;
nTest = 11579;

dist_first = zeros(nTest, nQuery, 'double');
fidin = fopen(first_dist_file);
for i = 1:nTest
    dist_line = fgetl(fidin);
    dist_line = str2num(dist_line);
    dist_first(i, 1:length(dist_line)) = dist_line; 
end
fclose(fidin);

final_dist = dist_first;

%% load ground truth index
maxgt = 256;
gt_index =  zeros(nQuery, maxgt);
fidin = fopen('../Data/VeRi/gt_index.txt');

for i = 1:nQuery
    gt_index_line = fgetl(fidin);
    gt_line = str2num(gt_index_line);
    for j = 1:size(gt_line, 2)
       gt_index(i, j) = gt_line(j); 
    end
end

maxjk = 256;
jk_index = zeros(nQuery, maxjk);
fidin = fopen('../Data/VeRi/jk_index.txt');
for i = 1:nQuery
    jk_index_line = fgetl(fidin);
    jk_line = str2num(jk_index_line);
    for j = 1:size(jk_line, 2)
       jk_index(i, j) = jk_line(j); 
    end
end
%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
CMC = zeros(nQuery, nTest);
r1 = 0; % rank 1 precision with single query
for k = 1:nQuery
    k;
    % load groud truth for each query (good and junk)
      good_index = reshape(gt_index(k,:), 1, []);
      good_index = good_index(good_index ~= 0);
      junk_index = reshape(jk_index(k,:), 1, []);
      junk_index = junk_index(junk_index ~= 0);
    tic
    score = final_dist(:, k);
    %score = dist_null(:, k);
%     score = dist_LOMO(:, k);
    [~, index] = sort(score, 'ascend');  % single query
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query

end
CMC = mean(CMC);
mean_AP = mean(ap);
top_1 = CMC(1);
top_5 = CMC(5);
%% print result
fprintf('single query:  mAP = %f,	r1 precision = %f,	r5 precision = %f\r\n', mean(ap), CMC(1), CMC(5));
%% plot CMC curves
figure;
s = 50;
CMC_curve = CMC;
plot(1:s, CMC_curve(:, 1:s));

end