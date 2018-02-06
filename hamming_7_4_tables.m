function [H, S_table] = hamming_7_4_tables()
H = [
    1 0 0 1 1 0 1;
    0 1 0 1 0 1 1;
    0 0 1 0 1 1 1];
% top row is binary representation of columns of H,
% bottom row is error locators. We sort the table to
% be able to address the error locations by index of
% numerical representation of syndrome vector.
S_table = [
    1 2 4 3 5 6 7;
    1 2 3 4 5 6 7];

S_table = sortrows(S_table', 1)';
S_table = S_table(2,:);
end