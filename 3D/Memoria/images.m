% images_FNO3D
images_name = 'images_FNO3D-1-1.mat';
source = load(images_name, 'source');
source = source.source;
target = load(images_name, 'target');
target = target.target;
J0 = load(images_name, 'J0');
J0 = J0.J0;

name = 'FNO3D-1-1';

sourceSlice1 = source(:, end/2, :);
targetSlice1 = target(:, end/2, :);
J0Slice1 = J0(:, end/2, :);

% rotate them
sourceSlice1 = permute(sourceSlice1, [2, 3, 1]);
targetSlice1 = permute(targetSlice1, [2, 3, 1]);
J0Slice1 = permute(J0Slice1, [2, 3, 1]);

% flip them
sourceSlice1 = fliplr(sourceSlice1);
targetSlice1 = fliplr(targetSlice1);
J0Slice1 = fliplr(J0Slice1);


sourceSlice2 = source(end/2, :, :);
targetSlice2 = target(end/2, :, :);
J0Slice2 = J0(end/2, :, :);

% rotate them
sourceSlice2 = permute(sourceSlice2, [3, 2, 1]);
targetSlice2 = permute(targetSlice2, [3, 2, 1]);
J0Slice2 = permute(J0Slice2, [3, 2, 1]);

% flip them
sourceSlice2 = fliplr(sourceSlice2);
targetSlice2 = fliplr(targetSlice2);
J0Slice2 = fliplr(J0Slice2);

sourceSlice2 = flipud(sourceSlice2);
targetSlice2 = flipud(targetSlice2);
J0Slice2 = flipud(J0Slice2);

sourceSlice3 = source(:, :, end/2);
targetSlice3 = target(:, :, end/2);
J0Slice3 = J0(:, :, end/2);

% rotate them
sourceSlice3 = permute(sourceSlice3, [2, 3, 1]);
targetSlice3 = permute(targetSlice3, [2, 3, 1]);
J0Slice3 = permute(J0Slice3, [2, 3, 1]);

% % flip them
sourceSlice3 = flipud(sourceSlice3);
targetSlice3 = flipud(targetSlice3);
J0Slice3 = flipud(J0Slice3);

sourceSlice1 = squeeze(sourceSlice1);
sourceSlice2 = squeeze(sourceSlice2);
sourceSlice3 = squeeze(sourceSlice3);

targetSlice1 = squeeze(targetSlice1);
targetSlice2 = squeeze(targetSlice2);
targetSlice3 = squeeze(targetSlice3);

J0Slice1 = squeeze(J0Slice1);
J0Slice2 = squeeze(J0Slice2);
J0Slice3 = squeeze(J0Slice3);



% figure;
% imshow(bigSourceImage, []);
% %title('NIREP', 'Interpreter', 'Latex', 'FontSize', fontsize);

% exportgraphics(gcf, 'imagesNIREP.png', 'BackgroundColor', 'none', 'ContentType', 'vector', 'Resolution', 900);

% figure;
% imshow(sourceSlice1, []);
% exportgraphics(gcf, 'oasis_source_coronal.png', 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(sourceSlice2, []);
exportgraphics(gcf, 'oasis_source_sagital.png', 'BackgroundColor', 'none', 'Resolution', 900);

% figure;
% imshow(sourceSlice3, []);
% exportgraphics(gcf, 'oasis_source_axial.png', 'BackgroundColor', 'none', 'Resolution', 900);

% figure;
% imshow(targetSlice1, []);
% exportgraphics(gcf, 'oasis_target_coronal.png', 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(targetSlice2, []);
exportgraphics(gcf, 'oasis_target_sagital.png', 'BackgroundColor', 'none', 'Resolution', 900);

% figure;
% imshow(targetSlice3, []);
% exportgraphics(gcf, 'oasis_target_axial.png', 'BackgroundColor', 'none', 'Resolution', 900);

% figure;
% imshow(J0Slice1, []);
% exportgraphics(gcf, strcat('oasis_', name, '_coronal.png'), 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(J0Slice2, []);
exportgraphics(gcf, strcat('oasis_', name, '_sagital.png'), 'BackgroundColor', 'none', 'Resolution', 900);

% figure;
% imshow(J0Slice3, []);
% exportgraphics(gcf, strcat('oasis_', name, '_axial.png'), 'BackgroundColor', 'none', 'Resolution', 900);


diff_ini = sourceSlice2 - targetSlice2;
figure;
imshow(diff_ini, []);
exportgraphics(gcf, strcat('nirep_sagital_diff_ini.png'), 'BackgroundColor', 'none', 'Resolution', 900);

diff2 = J0Slice2 - targetSlice2;
figure;
imshow(diff2, []);
exportgraphics(gcf, strcat('nirep_', name, '_sagital_diff.png'), 'BackgroundColor', 'none', 'Resolution', 900);




