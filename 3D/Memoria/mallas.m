
phi_name = '/home/epdiff/python/3D/Resultados/OASIS/FNO3D_1_1/phi_moving402_fixed403.mat';
phi = load(phi_name, 'phi');
phi = phi.phi;


nx = size(phi, 2);
ny = size(phi, 1);
nz = size(phi, 3);
hx = 1 / nx;
hy = 1 / ny;
hz = 1 / nz;
x = hx * [0:1:nx-1];
y = hy * [0:1:ny-1];
z = hz * [0:1:nz-1];
[X, Y, Z] = meshgrid( x, y, z );

phi(:,:,:,1) = X + phi(:,:,:,1);
phi(:,:,:,2) = Y + phi(:,:,:,2);
phi(:,:,:,3) = Z + phi(:,:,:,3);

phi(:,:,:,1) = phi(:,:,:,1) * nx;
phi(:,:,:,2) = phi(:,:,:,2) * ny;
phi(:,:,:,3) = phi(:,:,:,3) * nz;

% Abre una nueva figura
figure;
hold on;
step = 1;

% Sagittal

dim = size(phi);
disp(dim)


phi = fliplr(phi);
phi_s = squeeze(phi(round(dim(1)/2),:,:,[1,3]));
dim = size(phi_s);
disp(dim)

for i = 1 : step : dim(1)
    plot(phi_s(i,:,1), phi_s(i,:,2), 'b');
end

for j = 1 : step : dim(2)
    handle = plot(phi_s(:,j,1), phi_s(:,j,2), 'b');
end

axis off;
axis tight;

% Guarda la figura como un archivo jpg
saveas(gcf, 'oasis_sagittal_view_fno3d_1_1.jpg');
close(gcf); % Cierra la figura








