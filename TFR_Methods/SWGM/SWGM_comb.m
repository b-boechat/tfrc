function spec_prod_w = SWGM_comb(spects_matrix, beta)
% spectrogram_comb is a function intended to combine different
% time-frequency representations by a weighted geometric mean method.
% Inputs:
%   - spects_matrix: a matrix containing the aligned spectrograms stacked
%   in the third dimension
%   - Beta: parameter that controls the ammount of weight applied. Usable
%   values should be 0<Beta<1.

max_alpha = 20;

alphas = zeros(size(spects_matrix));

if beta == 0
    % Combining spectrograms using GM 
    spec_prod_w = geomean(spects_matrix,3);
else    
    % Computing alpha
    for spec_ind = 1:size(spects_matrix,3)

        % Performing geometric mean of the other spectrograms
        spects_matrix_aux = spects_matrix;
        spects_matrix_aux(:,:,spec_ind) = [];
        spects_geomean = geomean(spects_matrix_aux,3);

        alphas(:,:,spec_ind) = (spects_geomean./spects_matrix(:,:,spec_ind)).^beta;
    %     alphas(isnan(alphas(:,:,spec_ind))) = max(max(alphas(:,:,spec_ind)));
        alphas(isnan(alphas(:,:,spec_ind))) = max_alpha;
        alphas((alphas > max_alpha) | isnan(alphas)) = max_alpha;
    %     max(max(alphas(:,:,spec_ind)))
    %     return
    end
    
    % Combining spectrograms using the SWGM 
    spects_matrix_alphas = spects_matrix.^alphas;
    spec_prod_w = prod(spects_matrix_alphas,3).^(1./sum(alphas,3));
    
end

orig_energy = sum(sum(spects_matrix(:,:,1)));
comb_energy = sum(sum(spec_prod_w));

spec_prod_w = spec_prod_w*orig_energy/comb_energy;

end