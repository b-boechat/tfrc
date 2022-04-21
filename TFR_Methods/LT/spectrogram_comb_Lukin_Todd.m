function final_TFR = spectrogram_comb_Lukin_Todd(spectrograms_tensor, size_w_S, eta)
% spectrogram_comb is a function intended to combine different
% time-frequency representations by a weighted geometric mean method.
% Inputs:
%   - spects_matrix: a matrix containing the aligned spectrograms stacked
%   in the third dimension
%   - size_w_S(1) and size_w_S(2): number (odd) of rows and columns of the 2D window

% Vantagens: 
% - Poder combinar maior numero de representacoes sem perdas 
% - A amplitude dos picos segue a representacao com maior amplitude (maior esparsidade)

epsilon = 1e-10; % To avoid division by 0
size_w_S = ceil(size_w_S);

energy_save = 0; % This is used to save processing when there is no energy in a given frame

if mod(size_w_S(1), 2) == 0    
    size_w_S(1) = size_w_S(1) + 1;
%     fprintf('WARNING: size_w_S(1) must be an odd number! Using %d instead!\n', size_w_S(1));
end

if mod(size_w_S(2),2) == 0
    size_w_S(2) = size_w_S(2) + 1;
%     fprintf('WARNING: size_w_S(2) must be an odd number! Using %d instead!\n', size_w_S(2));
end

% Zero-padding the spectrograms to properly apply the 2D windows at the edges
max_size_w = size_w_S;

zp_specs = cat(1, zeros((max_size_w(1)-1)/2,size(spectrograms_tensor, 2), size(spectrograms_tensor,3)), spectrograms_tensor, ...
                   zeros((max_size_w(1)-1)/2,size(spectrograms_tensor, 2), size(spectrograms_tensor,3)));
zp_specs = cat(2, zeros(size(zp_specs, 1), (max_size_w(2)-1)/2, size(zp_specs,3)), zp_specs, ...
                   zeros(size(zp_specs, 1), (max_size_w(2)-1)/2, size(zp_specs,3)));

% Computing local smearing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

localSmearing_tensor =  zeros(size(spectrograms_tensor)); % matrix that stores local sparsity

energy_frames = sum(spectrograms_tensor(:,:,end), 1);

for col = (max_size_w(2)-1)/2 + 1 : size(spectrograms_tensor, 2) + (max_size_w(2)-1)/2

    if (~energy_save || energy_frames(col - (max_size_w(2)-1)/2) > 0)  % presence of some signal
    
        for row = (max_size_w(1)-1)/2 + 1 : size(spectrograms_tensor, 1) + (max_size_w(1)-1)/2
        
            for spec_ind = 1:size(spectrograms_tensor, 3)
                
                regionMatrix = zp_specs(row-(size_w_S(1)-1)/2:row+(size_w_S(1)-1)/2,...
                                                      col-(size_w_S(2)-1)/2:col+(size_w_S(2)-1)/2, spec_ind);
                windowedRegion = regionMatrix;
                localSmearing = compute_smearing(windowedRegion);
                localSmearing_tensor(row - (max_size_w(1)-1)/2, col - (max_size_w(2)-1)/2, spec_ind) = localSmearing;            
                
            end
            
        end
        
    end
    
end

% Mixing the TFRs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

weight = 1./(localSmearing_tensor.^eta + epsilon);

final_TFR = sum(spectrograms_tensor.*weight, 3)./sum(weight, 3);

orig_energy = sum(sum(spectrograms_tensor(:,:,1)));
comb_energy = sum(sum(final_TFR));

final_TFR = final_TFR*orig_energy/comb_energy;

end