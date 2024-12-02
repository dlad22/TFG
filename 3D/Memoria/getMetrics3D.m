gpuDevice(1);

reg_type = 'CN';
opt_type = 'GradientDescent';




dataset = 1;

if dataset == 0
    % NIREP
    patient_init = 2;
    patient_end = 16;
else
    % OASIS
    patient_init = 395;
    patient_end = 413;
end

% Para almacenar los resultados
DSC = zeros(patient_end - patient_init + 1, 1);
minJacs = zeros(patient_end - patient_init + 1, 1);
maxJacs = zeros(patient_end - patient_init + 1, 1);
negJacs = zeros(patient_end - patient_init + 1, 1);
i = 1;

for patient = patient_init : patient_end

    try
        if dataset == 0
            file = sprintf( '/home/epdiff/python/3D/Resultados/NIREP/FNO3D/phi_moving_1_fixed_%d.mat', patient );
            load( file );
            fprintf('Loaded %s\n', file);
            [meanDSC, Jacs, tNegJacs] = getAllMetricsNIREP(phi, patient);
        else
            file = sprintf('/home/epdiff/python/3D/Resultados/OASIS/FNO3D_1_1/Small/phi_moving%d_fixed%d.mat', patient, patient+1); % OASIS
            load( file );
            fprintf('Loaded %s\n', file);
            [meanDSC, Jacs, tNegJacs] = getAllMetricsOASIS(phi, patient);
        end
        
        DSC(i) = meanDSC;
        minJacs(i) = Jacs(2);
        maxJacs(i) = Jacs(1);
        negJacs(i) = tNegJacs;

        i = i + 1;
        
    catch ME
        % Mostrar el mensaje de error
        fprintf('Error in DSC: %s\n', ME.message);
        break;
    end
end


% Mostrar todos los DSC
for i = 1 : length(DSC)
    fprintf('%f\n', DSC(i));
end

mean_DSC = mean(DSC);

% % Mostrar todos los DSC
% for i = 1 : length(minJacs)
%     fprintf('%f\n', minJacs(i));
% end
mean_minJacs = mean(minJacs);

% % Mostrar todos los DSC
% for i = 1 : length(maxJacs)
%     fprintf('%f\n', maxJacs(i));
% end

mean_maxJacs = mean(maxJacs);
mean_negJacs = mean(negJacs);

% Calcular desviación tipica de meanDSC, minJacs, maxJacs
std_DSC = std(DSC);
std_minJacs = std(minJacs);
std_maxJacs = std(maxJacs);


fprintf('Mean DSC: %f\n', mean_DSC);
fprintf('Min Jacobian: %.15f\n', mean_minJacs);
fprintf('Max Jacobian: %.15f\n', mean_maxJacs);
fprintf('Negative Jacobians: %f\n', mean_negJacs);

fprintf('Std DSC: %f\n', std_DSC);
fprintf('Std Min Jacobian: %.15f\n', std_minJacs);
fprintf('Std Max Jacobian: %.15f\n', std_maxJacs);



% exit
