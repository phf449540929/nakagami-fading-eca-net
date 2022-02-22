function u=ldpc_encode(s,G)
    u=mod(s*G,2);
end