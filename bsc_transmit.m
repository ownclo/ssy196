function r = bsc_transmit(c, p)
r = mod(c + (rand(size(c)) < p), 2);
% only one error introduced
%r = mod(c + de2bi(2.^randi([0 size(c, 2)-1]), size(c, 2)), 2);
end
