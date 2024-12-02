

function [dscMean, Jacs, tNegJacs] = getAllMetricsNIREP(phi, patient)

    addpath( '/home/epdiff/matlab/Sources/MiaoMiaoZhang_FlashC/ODE_Solvers' );
    addpath( '/home/epdiff/matlab/Sources/Nifti' );
    addpath( '/home/epdiff/matlab/Sources/Vtk' );

    NIREP_dir = '/home/epdiff/python/3D/Data/NIREP/';
    
    % Load the images
    moving = load_vtk_float( sprintf( '%s/Subsample/NIREP_01-Sub.vtk', NIREP_dir ));
    fixed = load_vtk_float( sprintf( '%s/Subsample/NIREP_%02d-Sub.vtk', NIREP_dir, patient ));

    % Load the segmentations
    movingSegmentation = load_vtk_short( sprintf( '%s/Aligned/NIREP_01-SegReg.vtk', NIREP_dir ) );
    fixedSegmentation = load_vtk_short( sprintf( '%s/Aligned/NIREP_%02d-SegReg.vtk', NIREP_dir, patient ) );

    dimSegmentation = size(movingSegmentation);

    [nx, ny, nz] = size( fixed );

    % Obtain the transformation
    ww(:,:,:,1) = -phi(2:end-1,:,2:end-1,1);
    ww(:,:,:,2) = -phi(2:end-1,:,2:end-1,2);
    ww(:,:,:,3) = -phi(2:end-1,:,2:end-1,3);

    u1 = padarray(ww(:,:,:,1), [1, 0, 1], 'post');
    u2 = padarray(ww(:,:,:,2), [1, 0, 1], 'post');
    u3 = padarray(ww(:,:,:,3), [1, 0, 1], 'post');
        
    nxU = size(u1, 2);
    nyU = size(u1, 1);
    nzU = size(u1, 3);

    uu = zeros(nyU, nxU, nzU, 3, 'single');
    uu(:,:,:,1) = u1;
    uu(:,:,:,2) = u2;
    uu(:,:,:,3) = u3;

    fprintf('uu shape: %d %d %d %d\n', size(uu));

    u = uu;
    u = gpuArray(u);

    % Upsample the displacement
    fprintf('Upsampling the displacement...\n');
    dimTransformation = size(u);
    factor = dimSegmentation ./ dimTransformation(1:3);

    x = 0 : 1 : dimTransformation(2)-1;
    y = 0 : 1 : dimTransformation(1)-1;
    z = 0 : 1 : dimTransformation(3)-1;

    [X, Y, Z] = meshgrid(x, y, z);
    X = X * factor(1);
    Y = Y * factor(2);
    Z = Z * factor(3);

    x = 0 : 1 : dimSegmentation(2)-1;
    y = 0 : 1 : dimSegmentation(1)-1;
    z = 0 : 1 : dimSegmentation(3)-1;

    [XI, YI, ZI] = meshgrid(x, y, z);

    uu = zeros(dimSegmentation(1), dimSegmentation(2), dimSegmentation(3), 3, 'single', 'gpuArray');

    uu(:,:,:,1) = interp3(X, Y, Z, u(:,:,:,1) * factor(1) * nx, XI, YI, ZI, 'linear', 0);
    uu(:,:,:,2) = interp3(X, Y, Z, u(:,:,:,2) * factor(2) * ny, XI, YI, ZI, 'linear', 0);
    uu(:,:,:,3) = interp3(X, Y, Z, u(:,:,:,3) * factor(3) * nz, XI, YI, ZI, 'linear', 0);

    % Obtain the upsampled transformation
    iphi = zeros(dimSegmentation(1), dimSegmentation(2), dimSegmentation(3), 3, 'single', 'gpuArray');

    iphi(:,:,:,1) = XI - uu(:,:,:,1);
    iphi(:,:,:,2) = YI - uu(:,:,:,2);
    iphi(:,:,:,3) = ZI - uu(:,:,:,3);

    iphi = gather(iphi);

    % Obtain the warped segmentation
    fprintf('Warping the segmentation...\n');
    warpedSegmentation = interp3(XI, YI, ZI, movingSegmentation, iphi(:,:,:,1), iphi(:,:,:,2), iphi(:,:,:,3), 'nearest', 0);

    % Compute the metrics
    interVol = zeros(1,32);
    refVol = zeros(1,32);
    segVol = zeros(1,32);
    dsc = zeros(1,32);

    for label = 1 : 32
        aux = size(find(warpedSegmentation(find(fixedSegmentation(:) == label)) == label));
        interVol(1,label) = aux(1);
        aux = size(find(fixedSegmentation(:) == label));
        refVol(1,label) = aux(1);
        aux = size(find(warpedSegmentation(:) == label));
        segVol(1,label) = aux(1);
        
        dsc(label) = 2.0 * interVol(1,label) / (refVol(1,label) + segVol(1,label));
    end


    dscMean = mean(dsc);
    fprintf('DSC: %f\n', dscMean);

    Jaccard = dsc ./ ( 2 - dsc );

    % Jacobian

    % [nx, ny, nz, d] = size(uu);

    % fprintf('nx: %d ny: %d nz: %d\n', nx, ny, nz);
    % x = 1 : 1 : nx;
    % y = 1 : 1 : ny;
    % z = 1 : 1 : nz;

    % [X, Y, Z] = meshgrid( y, x, z );

    % fprintf('X shape %d %d %d\n', size(X));

    % XI = squeeze(X + u1);
    % YI = squeeze(Y + u2);
    % ZI = squeeze(Z + u3);

    % fprintf('XI shape %d %d %d\n', size(XI));

    % phi2(:,:,:,1) = XI;
    % phi2(:,:,:,2) = YI;
    % phi2(:,:,:,3) = ZI;

    % Jacobian

    % [nx, ny, nz, d] = size(warp);

    % fprintf('nx: %d ny: %d nz: %d d: %d\n', nx, ny, nz, d);
    % x = 1 : 1 : nx;
    % y = 1 : 1 : ny;
    % z = 1 : 1 : nz;

    % [X, Y, Z] = meshgrid( y, x, z );

    % fprintf('X shape %d %d %d\n', size(X));

    % XI = squeeze(X + warp(:,:,:,1));
    % YI = squeeze(Y + warp(:,:,:,2));
    % ZI = squeeze(Z + warp(:,:,:,3));

    % fprintf('XI shape %d %d %d\n', size(XI));

    % phi2(:,:,:,1) = XI;
    % phi2(:,:,:,2) = YI;
    % phi2(:,:,:,3) = ZI;



    fprintf('phi shape: %d %d %d %d\n', size(phi));
    nx = size(phi, 2);
    ny = size(phi, 1);
    nz = size(phi, 3);

    % phi = zeros(ny, nx, nz, 3, 'single');

    fprintf('nx: %d ny: %d nz: %d\n', nx, ny, nz);
    hx = 1 / nx;
    hy = 1 / ny;
    hz = 1 / nz;

    x = hx * [0:1:nx-1];
    y = hy * [0:1:ny-1];
    z = hz * [0:1:nz-1];

    [X, Y, Z] = meshgrid( x, y, z );

    fprintf('X dim: %d %d %d\n', size(X));
    fprintf('Y dim: %d %d %d\n', size(Y));
    fprintf('Z dim: %d %d %d\n', size(Z));

    fprintf('phi dim: %d %d %d %d\n', size(phi));

    phi_jac(:,:,:,1) = X + phi(:,:,:,1);
    phi_jac(:,:,:,2) = Y + phi(:,:,:,2);
    phi_jac(:,:,:,3) = Z + phi(:,:,:,3);

    phi_jac(:,:,:,1) = phi_jac(:,:,:,1) * nx;
    phi_jac(:,:,:,2) = phi_jac(:,:,:,2) * ny;
    phi_jac(:,:,:,3) = phi_jac(:,:,:,3) * nz;

    Jac = Jacobian(phi_jac);

    climit = 6;
    Jac = Jac(climit:end-climit, climit:end-climit, climit:end-climit);
    Jacs = [];
    tNegJacs = [];

    Jacs = [Jacs; max(Jac(:)), min(Jac(:))];


    tNegJacs = [tNegJacs; sum(Jac(:) < 0)];

    % % Mostrar tamaño de Jacs
    % fprintf('Jacs size: %d %d\n', size(Jacs));
    % fprintf('tNegJacs size: %d \n', sum(Jac(:)));

    display( Jacs )
    display( tNegJacs )

end
