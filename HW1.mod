param K;
param a0{0..1};
param v0{0..1};
param base0{0..1};
param base1{0..1};
param base2{0..1};
param base3{0..1};

var h >= 0;
#var h{0..3} >= 0; 
var p{0..4*K,0..1}; # initially 0
var v{0..4*K,0..1}; # initially v0
var a{0..4*K,0..1}; # initially a0

# objective function
#minimize minTime: K*(sum {i in 0..3} h[i]);
minimize minTime: 4*K*h;

# initial values for acceleration and velocity
#subject to initialAcc {i in 0..1} : a[0,i] = a0[i];
subject to initialVel {i in 0..1} : v[0,i] = v0[i];

# must pass by the bases
subject to initialBase {i in 0..1} : p[0*K,i] = base0[i];
subject to secondBase  {i in 0..1} : p[1*K,i] = base1[i];
subject to thirdBase   {i in 0..1} : p[2*K,i] = base2[i];
subject to fourthBase  {i in 0..1} : p[3*K,i] = base3[i];
subject to loopBase    {i in 0..1} : p[4*K,i] = base0[i];

# physical constraints
#subject to velocity0 {k in 0..K-1, i in 0..1}: v[k+1,i] = v[k,i] + h[0]*a[k,i];
#subject to velocity1 {k in K..2*K-1, i in 0..1}: v[k+1,i] = v[k,i] + h[1]*a[k,i];
#subject to velocity2 {k in 2*K..3*K-1, i in 0..1}: v[k+1,i] = v[k,i] + h[2]*a[k,i];
#subject to velocity3 {k in 3*K..4*K-1, i in 0..1}: v[k+1,i] = v[k,i] + h[3]*a[k,i];
subject to velocity {k in 0..4*K-1, i in 0..1}: v[k+1,i] = v[k,i] + h*a[k,i];

#subject to position0 {k in 0..K-1, i in 0..1}: p[k+1,i] = p[k,i] + h[0]*v[k,i];
#subject to position1 {k in K..2*K-1, i in 0..1}: p[k+1,i] = p[k,i] + h[1]*v[k,i];
#subject to position2 {k in 2*K..3*K-1, i in 0..1}: p[k+1,i] = p[k,i] + h[2]*v[k,i];
#subject to position3 {k in 3*K..4*K-1, i in 0..1}: p[k+1,i] = p[k,i] + h[3]*v[k,i];
subject to position {k in 0..4*K-1, i in 0..1}: p[k+1,i] = p[k,i] + h*v[k,i];

# acceleration upper bound
subject to notFaster {k in 0..4*K}: a[k,0]^2 + a[k,1]^2 <= 100

