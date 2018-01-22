%% Hamming Decoder
% 'Hamming' is important here because it is assumed that S_table
% is built for one-bit errors.
% assume H is in systematic form
% assume H is a proper parity-check matrix for a hamming code
% r is a received word from channel.
% S_table is a table from syndrome (in decimal form) to one-error location
% c_hat is the estimated _code_ word
% u_hat is the estimated _information_ word
function [c_hat, u_hat] = syndrome_decode_hamming(H, S_table, r)
s = mod(r*H.', 2);
% could be replaced with a precomputed table from syndromes to error positions.
% errloc = ismember(H.', s, 'rows').';
% c_hat = mod(r + errloc, 2);
c_hat = r;
s = bi2de(s);
if (s ~= 0)
    errloc = S_table(s);
    c_hat(errloc) = mod(c_hat(errloc)+1, 2);
end

u_hat = c_hat(size(H, 1) + 1 : end);
end
