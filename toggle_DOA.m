%% Source code for domain of attraction estimate by Ayush Pandey
% BSD 3-Clause License

% Copyright (c) 2020, Ayush Pandey
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
% 
% * Neither the name of the copyright holder nor the names of its
%   contributors may be used to endorse or promote products derived from
%   this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


% ---- 

%% # Toggle switch model defined as functions f and fs (check the end of
% this script)

%% Computer equilibrium points

alpha1 = 1.3;
alpha2 = 1;
beta = 3;
gamma = 10;
options = optimoptions('fsolve','Display','none');
params = [alpha1, alpha2, beta, gamma];
% # Find equilibrium points (two stable, one unstable) E_i
n = 2;
x0 = [0; -10];
E1 = fsolve(@(x) f(x, params), x0, options);
% disp('Unstable equilibrium is');
E1;
x0 = [100; 0];
E2 = fsolve(@(x) f(x, params), x0, options);
% disp('Stable equilibrium #1 is');
E2;
x0 = [0;100];
E3 = fsolve(@(x) f(x, params), x0, options);
% disp('Stable equilibrium #2 is');
E3;

%% Compute equilbirium points for fs (the function f with equilibrium shifted to origin)

params1 = [alpha1, alpha2, beta, gamma, E1(1), E1(2)];
params2 = [alpha1, alpha2, beta, gamma, E2(1), E2(2)];
params3 = [alpha1, alpha2, beta, gamma, E3(1), E3(2)];
% # Find equilibrium points (two stable, one unstable) E_i
x0 = [0;0];
E1s = fsolve(@(x) fs(x, params1), x0, options);
% disp('Unstable equilibrium is ')
E1s;
x0 = [100;0];
E2s = fsolve(@(x) fs(x, params2), x0, options);
% disp('Stable equilibrium #1 is')
E2s;
x0 = [0;100];
E3s = fsolve(@(x) fs(x, params3), x0, options);
% disp('Stable equilibrium #2 is')
E3s;

%% Check that E1s etc are all zero! We set that next. 

E1s = [0;0];
E2s = [0;0];
E3s = [0;0];

% To compute linearized dynamics, just compute the Jacobian as follows:
% # Linearized dynamics
% 
% f_ode = @(xs) fs(xs, params1);
% A1 = jacobian(f_ode, E1s);
% disp('The linearization at the unstable equilibrium is A1 with eigen values A1eig.')

% The linearized dynamics for each equilibrium point of the toggle switch
% are the following 
A1 = [-1, -1.09135581;-1.94292451,-1];
A2 = [-1,-0.01793196;-0.48652444,-1];
A3 = [-1,-0.99127747;-0.25173376,-1];

%% For linearized dynamics compute P using the Lyapunov equation
% # For A1
I = eye(2);
P1 = lyap(A1', I);
% # For A2
P2 = lyap(A2', I);
% # For A3
P3 = lyap(A3', I);
% 
%% Compute logarithm norm with the P computed above
% Check weighted_mu function defined at the end of the script
%     
muA1 = weighted_mu(A1, P1);
muA2 = weighted_mu(A2, P2);
muA3 = weighted_mu(A3, P3);

% Either compute a d based on A and P and see if it works
% or set the d-invariant set manually. (Here we do it manually, for best ).

d2 = compute_d(A2,P2);
d3 = compute_d(A3,P3);

d2 = 0.1;
d3 = 0.8;

%% Compute radius of ball in d-invariant set
% Check Definition 4 in the paper 
zeta2 = -1 * (exp(d2*muA2) - 1);
zeta3 = -1 * (exp(d3*muA3) - 1);
dinv2 = (d2 * muA2^2) / zeta2;
dinv3 = (d3 * muA3^2) / zeta3;
r2 = dinv2 +0.05;
r3 = dinv3-0.2;

%% Define the C as in equation (28)
% Note that to get the best estimate for the DOA, you can tune for a
% higher C values for each equilibrium point
% fmincon can be used to maximize for this objective. 

C2 = 0.2;
C3 = 1.7;

%% Analytically compute W(x) - the Lyapunov function

x1s = linspace(-2,5,100);
x2s = linspace(-2,5,100);

pointsx = [];
pointsy = [];
pointsx3 = [];
pointsy3 = [];

% Symbolically write down the toggle switch model so that we can perform
% integration to compute Wx as given by the formula in the paper equation
% (30)
x = sym('x',[2,1]);
syms t
f2 = [alpha1/(1 + (x(2) + E2(2))^beta) - (x(1) + E2(1));alpha2/(1 + (x(1) + E2(1))^gamma) - (x(2) + E2(2))];
f3 = [alpha1/(1 + (x(2) + E3(2))^beta) - (x(1) + E3(1));alpha2/(1 + (x(1) + E3(1))^gamma) - (x(2) + E3(2))];

% The function to integrate (see equation 30)
int_fun2 = (x + t*f2)'*P2*(x+t*f2);
int_fun3 = (x + t*f3)'*P3*(x+t*f3);


Wx2 = int(int_fun2, t, 0, d2);
Wx3 = int(int_fun3, t, 0, d3);
gradWx2 = gradient(Wx2, x);
gradWx3 = gradient(Wx3, x);
Wdot2 = gradWx2' * f2;
Wdot3 = gradWx3' * f3;


%% Search algorithm to find DOAs:
for i = 1:length(x1s)
    x1 = x1s(i);
    for j = 1:length(x2s)
        x2 = x2s(j);
        if double(subs(Wx2, x, [x1; x2])) - C2 <= 0 && double(subs(Wdot2, x, [x1;x2])) <= 0
            if Sd([x1;x2], E2, r2) <= 0
                pointsx = [pointsx, x1];
                pointsy = [pointsy, x2];
            end
        end
        if double(subs(Wx3, x, [x1; x2])) - C3 <= 0 && double(subs(Wdot3, x, [x1;x2])) <= 0
            if Sd([x1;x2], E3, r3) <= 0
                pointsx3 = [pointsx3, x1];
                pointsy3 = [pointsy3, x2];
            end
        end   
    end
end

%% Find boundary from the results of the search algorithm
pointsx = pointsx';
pointsy = pointsy';
pointsx3 = pointsx3';
pointsy3 = pointsy3';
% sols_x[i] = sol(1);
% sols_y[i] = sol(2);
k = boundary(pointsx,pointsy);
k3 = boundary(pointsx3,pointsy3);

%% Plot the results
hold on
grid on
plot(pointsx(k),pointsy(k),'b','LineWidth',2);
plot(pointsx3(k3),pointsy3(k3),'r','LineWidth',2);
% plot the d-invariant sets (uncomment, if needed)
% plot_circle(E2(1),E2(2),r2)
% plot_circle(E3(1),E3(2),r3)
% scatter(pointsx, pointsy)
plot(E1(1),E1(2),'kx')
plot(E2(1), E2(2), 'bo')
plot(E3(1), E3(2), 'ro')

%% All functions here
function res = Sd(x,E,r)
    % This function checks whether the given point (vector) x is inside a
    % ball of radius r with center at E. If inside, res will be negative. 
    res = norm(x - E) - r;
end

function plot_circle(x,y,r)
    % (Code taken from MathWorks: 
    % https://se.mathworks.com/matlabcentral/answers/347942-filled-circles-with-different-colors)
    
    %x and y are the coordinates of the center of the circle
    %r is the radius of the circle
    %0.01 is the angle step, bigger values will draw the circle faster but
    %you might notice imperfections (not very smooth)
    ang=0:0.01:2*pi; 
    xp=r*cos(ang);
    yp=r*sin(ang);
    plot(x+xp,y+yp, 'k--', 'LineWidth',0.5);
end

function mu = weighted_mu(A, P)
    % Computes the logarithmic norm induced by the 2-norm 
    % as defined in Section I-A in the paper:
    % Doban, Alina Ionela, and Mircea Lazar. 
    % "Computation of Lyapunov functions for nonlinear differential 
    % equations via a Massera-type construction." IEEE Transactions 
    % on Automatic Control 63.5 (2017): 1259-1272.
    wp = sqrtm(P)*A/sqrtm(P);
    mx = (1/2) * ( wp + wp' );
    mu = max(eig(mx));
end


function d = compute_d(A, P)
    dcs = linspace(5,0.1,100);
    flag = false;
    for i = 1:length(dcs)
        dc = dcs(i);
        wp = sqrtm(P)*expm(dc*A)/sqrtm(P);
        mx = (1/2) * (wp + wp');
        if max(eig(mx)) < 1
            d = dc;
            flag = true;
        end
    end
    if ~flag
        disp('No d found, terminating.')
    end
end

%% Load model here
function y = f(x,params)
    % The toggle switch model
    alpha1 = params(1);
    alpha2 = params(2);
    beta = params(3);
    gamma = params(4);
    pt_dot =  alpha1/(1 + x(2)^beta) - x(1);
    pl_dot = alpha2/(1 + x(1)^gamma) - x(2);
    y = [pt_dot; pl_dot];
end


function y = fs(x,params)
    % The toggle switch model with updated equilibrium point to origin
    alpha1 = params(1);
    alpha2 = params(2);
    beta = params(3);
    gamma = params(4);
    E(1) = params(5);
    E(2) = params(6);
    pt_dot =  alpha1/(1 + (x(2) + E(2))^beta) - (x(1) + E(1));
    pl_dot = alpha2/(1 + (x(1) + E(1))^gamma) - (x(2) + E(2));
    y = [pt_dot;pl_dot];
end
