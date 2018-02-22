%% Binary search for a monotonic function.
% Given that testfun is defined on a range from low to high and
% is false from low to some point, then it is true from this point
% to high, return a range from low to high where the distance from
% low to high is smaller than delta, for which low is still false
% and high is already true.
function [low, high] = binsearch(testfun, from, to, precision)
low = from;
high = to;

while (high - low > precision)
    middle = low + (high - low) / 2;
    if testfun(middle)
        high = middle;
    else
        low = middle;
    end
end

end