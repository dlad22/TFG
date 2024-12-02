filename = '/home/epdiff/python/2D/ZhangLDDMM/images_fno2dt_1_5.mat';

source = load(filename, 'source');
source = source.source;
target = load(filename, 'target');
target = target.target;
J0 = load(filename, 'J0');
J0 = J0.J0;

name = '1-5';

diff_ini = source - target;
diff = J0 - target;


figure;
imshow(source, []);
exportgraphics(gcf, strcat('bulleyes_', name, '_source.png'), 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(target, []);
exportgraphics(gcf, strcat('bulleyes_', name, '_target.png'), 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(J0, []);
exportgraphics(gcf, strcat('bulleyes_', name, '_J0_fno2dt.png'), 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(diff_ini, []);
exportgraphics(gcf, strcat('bulleyes_', name, '_diff_ini.png'), 'BackgroundColor', 'none', 'Resolution', 900);

figure;
imshow(diff, []);
exportgraphics(gcf, strcat('bulleyes_', name, '_diff_fno2dt.png'), 'BackgroundColor', 'none', 'Resolution', 900);


