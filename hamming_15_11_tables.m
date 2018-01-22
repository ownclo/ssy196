function [H, S_table] = hamming_15_11_tables()
H = [
    1 0 0 0 0 0 0 0 1 1 1 1 1 1 1;
    0 1 0 0 0 1 1 1 0 0 0 1 1 1 1;
    0 0 1 0 1 0 1 1 0 1 1 0 0 1 1;
    0 0 0 1 1 1 0 1 1 0 1 0 1 0 1];

% top row is binary representation of columns of H,
% bottom row is error locators. We sort the table to
% be able to address the error locations by index of
% numerical representation of syndrome vector.
S_table = [
    1 2 4 8 12 10 6 14  9  5 13  3 11  7 15;
    1 2 3 4  5  6 7  8  9 10 11 12 13 14 15];
S_table = sortrows(S_table', 1)';
S_table = S_table(2,:);
end
