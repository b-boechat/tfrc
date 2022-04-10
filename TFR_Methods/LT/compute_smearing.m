function energyCompaction = compute_smearing(S)

energyCompaction = zeros(1,size(S,3));

for tfrInd = 1:size(S,3)
        vector = reshape(S(:,:,tfrInd), 1, size(S(:,:,tfrInd),1)*size(S(:,:,tfrInd),2));
        sortedVector = sort(vector, 'descend');
        N = length(sortedVector);
        indVector = 1:N;
        firstMoment = sum(sortedVector.*indVector);

        energyCompaction(tfrInd) = firstMoment/(sqrt(sum(sortedVector)) + 10^(-6));
end

end