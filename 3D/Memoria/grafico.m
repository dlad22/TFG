


% Datos de DSC para los conjuntos de datos de NiRep y OASIS
nirep_ode_small = [0.426732, 0.385402, 0.398186, 0.376678, 0.361743, 0.365199, 0.384695, 0.388054, 0.384664, 0.460216, 0.427487, 0.432320, 0.372801, 0.370031, 0.409414];
nirep_fno3d_small = [0.432459, 0.372905, 0.373163, 0.388549, 0.379887, 0.374524, 0.385635, 0.391266, 0.381286, 0.463418, 0.431757, 0.424857, 0.380014, 0.375778, 0.410182];
nirep_0_1_small = [0.448069, 0.417467, 0.408138, 0.395582, 0.399115, 0.412777, 0.411957, 0.423663, 0.406089, 0.452869, 0.447309, 0.449264, 0.394279, 0.388015, 0.429456];
nirep_1_1_small = [0.423585, 0.382831, 0.382905, 0.375416, 0.376642, 0.367415, 0.378599, 0.386668, 0.375908, 0.458835, 0.424085, 0.423670, 0.376555, 0.375760, 0.403073];

nirep_ode_large = [0.552919, 0.499586, 0.525441, 0.525840, 0.518890, 0.516741, 0.536224, 0.515099, 0.564075, 0.556054, 0.531734, 0.530317, 0.505352, 0.543365, 1.53];
nirep_fno3d_large = [0.440358, 0.416938, 0.488219, 0.394921, 0.423640, 0.387499, 0.428081, 0.401811, 0.462279, 0.522363, 0.472609, 0.468765, 0.434168, 0.408740, 0.469306];
nirep_0_1_large = [0.456992, 0.394719, 0.487829, 0.394633, 0.419980, 0.397704, 0.396034, 0.391745, 0.436149, 0.459032, 0.438703, 0.420676, 0.411851, 0.393107, 0.450134];
nirep_1_1_large = [0.555348, 0.505962, 0.491546, 0.524047, 0.526969, 0.518930, 0.525781, 0.543665, 0.515952, 0.564261, 0.554613, 0.538219, 0.531329, 0.506830, 0.520375];

oasis_ode_small = [0.626724, 0.569174, 0.595679, 0.531792, 0.611755, 0.574068, 0.486925, 0.560418, 0.628050, 0.543864, 0.593297, 0.580658, 0.546429, 0.673634, 0.578815, 0.642135, 0.618592, 0.600751, 0.600593];
oasis_fno3d_small = [0.617132, 0.578796, 0.595908, 0.531655, 0.599187, 0.571367, 0.468233, 0.570603, 0.619507, 0.541406, 0.593134, 0.571943, 0.540897, 0.665931, 0.581961, 0.636914, 0.617771, 0.605709, 0.600397];
oasis_0_1_small = [0.606275, 0.576141, 0.596476, 0.526413, 0.593246, 0.558709, 0.471816, 0.564810, 0.609634, 0.534110, 0.595266, 0.558863, 0.529941, 0.654305, 0.584370, 0.626343, 0.607869, 0.601924, 0.590704];
oasis_1_1_small = [0.618959, 0.573754, 0.599260, 0.527972, 0.604294, 0.570308, 0.477304, 0.565882, 0.622461, 0.539773, 0.595827, 0.574706, 0.543438, 0.663891, 0.580975, 0.636540, 0.614030, 0.608722, 0.600557];

oasis_ode_large = [0.692944, 0.650671, 0.699526, 0.614493, 0.682576, 0.683971, 0.603169, 0.674084, 0.740213, 0.651016, 0.653367, 0.667289, 0.638239, 0.745633, 0.735921, 0.718321, 0.699039, 0.684258, 0.679869];
oasis_fno3d_large = [0.602245, 0.649702, 0.699500, 0.618483, 0.690203, 0.690129, 0.588750, 0.684096, 0.726492, 0.654109, 0.669602, 0.665652, 0.628423, 0.743197, 0.655990, 0.724974, 0.705971, 0.703581, 0.643080];
oasis_0_1_large = [0.534221, 0.607856, 0.632901, 0.608099, 0.655217, 0.663717, 0.546246, 0.673453, 0.687310, 0.638534, 0.669531, 0.641369, 0.588654, 0.722419, 0.558922, 0.711085, 0.696641, 0.681910, 0.573946];
oasis_1_1_large = [0.694029, 0.658347, 0.707269, 0.607745, 0.678955, 0.679605, 0.597288, 0.675737, 0.737304, 0.647015, 0.659152, 0.666017, 0.639879, 0.744197, 0.735512, 0.718804, 0.702331, 0.693796, 0.684145];

% Transponer los datos para que cada columna represente un conjunto de datos
nirep_ode_large = nirep_ode_large';
nirep_ode_small = nirep_ode_small';
nirep_fno3d_large = nirep_fno3d_large';
nirep_fno3d_small = nirep_fno3d_small';
nirep_0_1_large = nirep_0_1_large';
nirep_0_1_small = nirep_0_1_small';
nirep_1_1_large = nirep_1_1_large';
nirep_1_1_small = nirep_1_1_small';

oasis_ode_large = oasis_ode_large';
oasis_ode_small = oasis_ode_small';
oasis_fno3d_large = oasis_fno3d_large';
oasis_fno3d_small = oasis_fno3d_small';
oasis_0_1_large = oasis_0_1_large';
oasis_0_1_small = oasis_0_1_small';
oasis_1_1_large = oasis_1_1_large';
oasis_1_1_small = oasis_1_1_small';

% Combinar los datos en una matriz
% data = [nirep_ode_small, nirep_fno3d_small, nirep_0_1_small, nirep_1_1_small, nirep_ode_large, nirep_fno3d_large, nirep_0_1_large, nirep_1_1_large];
data = [oasis_ode_small, oasis_fno3d_small, oasis_0_1_small, oasis_1_1_small, oasis_ode_large, oasis_fno3d_large, oasis_0_1_large, oasis_1_1_large];

% Crear el gráfico de cajas y bigotes
fig = figure('Visible', 'off');
boxplot(data, 'Labels', {'FLASH', 'FNO3D', 'FNO3D [0,1]', 'FNO3D [-1,1]', 'FLASH', 'FNO3D', 'FNO3D [0,1]', 'FNO3D [-1,1]'}); %, 'Labels', {'ODE Small', 'FNO3D Small', '1_1 Small', '0_1 Small', 'ODE Large', 'FNO3D Large', '1_1 Large', '0_1 Large'});
% title('Gráfico de Cajas y Bigotes para 8 Conjuntos de Datos');
% xlabel('Conjuntos de Datos');
ylabel('DSC (%)');

% Definir los colores para cada caja
colors = [
    1 0 0; % Rojo
    0 1 0; % Verde
    0 0 1; % Azul
    1 1 0; % Amarillo
    1 0 0; % Rojo
    0 1 0; % Verde
    0 0 1; % Azul
    1 1 0; % Amarillo
];

% Obtener los objetos de las cajas y modificar sus colores
hBox = findobj(gca, 'Tag', 'Box');
hWhisker = findobj(gca, 'Tag', 'Whisker');

for j = 1:length(hBox)
    fprintf('j: %d\n', j);
    disp(colors(j,:));
    % Colores para las cajas
    patch(get(hBox(j), 'XData'), get(hBox(j), 'YData'), colors(j,:), 'FaceColor', 'none', 'EdgeColor', colors(j,:), 'LineWidth', 1);
end



% Añadir una leyenda para cada conjunto de datos
legend({'FLASH', 'FNO3D', 'FNO3D [0,1]', 'FNO3D [-1,1]'}, ...
       'Location', 'north', 'Orientation', 'horizontal');



% Establecer el rango del eje Y
% ylim([0.35, 0.6]);
ylim([0.45, 0.8]);

% Establecer las marcas (ticks) del eje Y
% yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6]);
yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]);

% Establecer los ticks en el eje X sin mostrar las etiquetas
xticks([4.5]);  % Define las posiciones de los ticks
xticklabels('');  % Elimina las etiquetas de los ticks

% Añadir texto en una posición deseada
% text(2, 0.1, '90 x 105 x 90');
% text(6, 0.1, '180 x 210 x 180');

text(2, 0.1, '80 x 112 x 96');
text(6, 0.1, '160 x 224 x 192');

% Guardar el gráfico en un archivo
saveas(fig, 'grafico_cajas_colores.png'); % Guardar como PNG

% Cerrar la figura
close(fig);
