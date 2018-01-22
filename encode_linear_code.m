function c = encode_linear_code(G, word)
c = mod(word * G, 2);
end
