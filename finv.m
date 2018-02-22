function x = finv(fx, y, from, to)
x = fminbnd(@(x) abs(fx(x) - y), from, to);
end