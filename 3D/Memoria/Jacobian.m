function Jac = Jacobian(phi)

[D1, D2, D3] = gradient( phi(:,:,:,1) );
[D4, D5, D6] = gradient( phi(:,:,:,2) );
[D7, D8, D9] = gradient( phi(:,:,:,3) );

Jac = D1.*D5.*D9 - D1.*D6.*D8 - D2.*D4.*D9 + D2.*D6.*D7 + D3.*D4.*D8 - D3.*D5.*D7;

