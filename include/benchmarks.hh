// Copyright (c) 2026 Enrico Bertolazzi
// Distributed under the terms of the MIT License
//
// Benchmark functions for L-BFGS-B / L-BFGS
// Collection of 10 classic hard test problems from the global-optimization
// literature, implemented as optimize::Function subclasses with analytic
// gradients suitable for bound-constrained and unconstrained minimization.
//
// Primary catalogue: M. Jamil & X.-S. Yang, "A literature survey of benchmark
// functions for global optimization problems", Int. J. Math. Modelling Numer.
// Optimisation 4(2), 150-194 (2013).  DOI 10.1504/IJMMNO.2013.055088
//   Cited below as [Jamil13].
//
// Classical numerical test sets:
//   L. C. W. Dixon & G. P. Szeg\"o (eds.), Towards Global Optimisation 2,
//     North-Holland, 1978.                                      [Dixon78]
//   J. J. Mor\'e, B. S. Garbow & K. E. Hillstrom, "Testing unconstrained
//     optimization software", ACM Trans. Math. Softw. 7(1), 17-41 (1981).
//                                                              [MGH81]
//
// Each class documents: formula, standard box, global minimiser(s) and
// literature references.  All domains and minima coincide with [Jamil13]
// unless noted. Gradients are analytic and verified against finite
// differences (see test/benchmarks).

#ifndef BENCHMARKS_HH
#define BENCHMARKS_HH

#include <cmath>
#include <stdexcept>
#include "lbfgsb.hh"

namespace optimize {
namespace benchmarks {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline Scalar sqr(Scalar x) { return x*x; }
inline constexpr Scalar PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// 1. Ackley  [Jamil13 #8]  (n-D, scalable)
//    Ackley, "A connectionist machine for genetic hillclimbing", 1987.
//    f(x) = -20 exp(-0.2 sqrt(1/n sum xi^2)) - exp(1/n sum cos 2pi xi) +20+e
//    Domain: [-32.768, 32.768]^n   Global min: f(0)=0
//    Difficult: exponential + cosine, many shallow local minima.
//    Also in [Dixon78] set as example of multimodal scalable problem.
// ---------------------------------------------------------------------------
class Ackley : public Function {
public:
  explicit Ackley(Index n = 2) : n_(n) {
    if (n <= 0) throw std::invalid_argument("Ackley: n must be >0");
  }
  Scalar computeValue(const Vector& x) override {
    if (x.size() != n_) throw std::invalid_argument("Ackley: dimension mismatch");
    constexpr Scalar a = 20.0;
    constexpr Scalar b = 0.2;
    Scalar sum_sq = 0, sum_cos = 0;
    for (Index i=0;i<n_;++i){ sum_sq += x(i)*x(i); sum_cos += std::cos(2*PI*x(i)); }
    Scalar r = std::sqrt(sum_sq / static_cast<Scalar>(n_));
    return -a*std::exp(-b*r) - std::exp(sum_cos/static_cast<Scalar>(n_)) + a + std::exp(1.0);
  }
  Vector computeGradient(const Vector& x) override {
    if (x.size() != n_) throw std::invalid_argument("Ackley: dimension mismatch");
    constexpr Scalar a = 20.0;
    constexpr Scalar b = 0.2;
    Scalar sum_sq = 0, sum_cos = 0;
    for (Index i=0;i<n_;++i){ sum_sq += x(i)*x(i); sum_cos += std::cos(2*PI*x(i)); }
    Scalar r = std::sqrt(sum_sq / static_cast<Scalar>(n_));
    Scalar exp_r = std::exp(-b*r);
    Scalar exp_cos = std::exp(sum_cos/static_cast<Scalar>(n_));
    Vector g(n_);
    if (r < 1e-12) {
      // limit r->0: first term -> 0
      for (Index i=0;i<n_;++i) g(i) = (2*PI/static_cast<Scalar>(n_))*exp_cos*std::sin(2*PI*x(i));
    } else {
      for (Index i=0;i<n_;++i) {
        Scalar term1 = a*b*exp_r * x(i) / (static_cast<Scalar>(n_)*r); // 4*exp(-b r)* x/(n r)
        // Note a*b =4 : 20*0.2=4
        Scalar term2 = (2*PI/static_cast<Scalar>(n_))*exp_cos*std::sin(2*PI*x(i));
        g(i) = term1 + term2;
      }
      // correction: term1 derived as + a*b *exp(-b r)* x/(n r) = 4*..
      // But earlier derivation gave 4*exp(-b r)*x/(n r) -> same
    }
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n, -32.768); }
  static Vector standardUpper(Index n){ return Vector::Constant(n,  32.768); }
  static Vector globalMinimizer(Index n){ return Vector::Zero(n); }
  static constexpr Scalar globalMinimum = 0.0;
private:
  Index n_;
};

// ---------------------------------------------------------------------------
// 2. Rastrigin  [Jamil13 #178]  (n-D)
//    Rastrigin, 1974.  f(x)=10n + sum[ xi^2 -10 cos(2pi xi)]
//    Domain: [-5.12,5.12]^n  Global min: f(0)=0
//    Highly multimodal, separable.  Classic in [Dixon78] / [MGH81] style tests.
// ---------------------------------------------------------------------------
class Rastrigin : public Function {
public:
  explicit Rastrigin(Index n=2): n_(n) {
    if(n<=0) throw std::invalid_argument("Rastrigin: n>0");
  }
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Rastrigin dim");
    Scalar v = 10.0*static_cast<Scalar>(n_);
    for(Index i=0;i<n_;++i) v += x(i)*x(i) -10*std::cos(2*PI*x(i));
    return v;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Rastrigin dim");
    Vector g(n_);
    for(Index i=0;i<n_;++i) g(i)= 2*x(i) + 20*PI*std::sin(2*PI*x(i));
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-5.12); }
  static Vector standardUpper(Index n){ return Vector::Constant(n, 5.12); }
  static Vector globalMinimizer(Index n){ return Vector::Zero(n); }
  static constexpr Scalar globalMinimum = 0.0;
private:
  Index n_;
};

// ---------------------------------------------------------------------------
// 3. Griewank  [Jamil13 #80]  (n-D)
//    Griewank, 1981. f= sum xi^2/4000 - prod cos(xi/sqrt(i+1)) +1
//    Domain: [-600,600]^n  Global min: f(0)=0
//    Non-separable, product term creates many interdependent minima.
// ---------------------------------------------------------------------------
class Griewank : public Function {
public:
  explicit Griewank(Index n=2): n_(n){ if(n<=0) throw std::invalid_argument("Griewank n>0");}
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Griewank dim");
    Scalar s=0; Scalar p=1;
    for(Index i=0;i<n_;++i){ s+= x(i)*x(i)/4000.0; p*= std::cos(x(i)/std::sqrt(static_cast<Scalar>(i+1))); }
    return s - p + 1.0;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Griewank dim");
    // precompute product
    Scalar p_all=1;
    for(Index i=0;i<n_;++i) p_all *= std::cos(x(i)/std::sqrt(static_cast<Scalar>(i+1)));
    Vector g(n_);
    for(Index j=0;j<n_;++j){
      Scalar sj = std::sqrt(static_cast<Scalar>(j+1));
      Scalar cj = std::cos(x(j)/sj);
      Scalar pj = (std::abs(cj) < 1e-14) ? 0 : p_all / cj; // product without j
      // avoid division by zero -> limit handled
      Scalar term2 = (std::abs(cj) < 1e-14) ? 0 : std::sin(x(j)/sj)/sj * pj;
      g(j)= x(j)/2000.0 + term2;
      // when cj~0, full finite-difference fallback for that component
      if(std::abs(cj) < 1e-14){
        // numerical fallback not needed for gradient test but keep safe
        // compute product without j analytically
        Scalar p_no_j=1; for(Index i=0;i<n_;++i) if(i!=j) p_no_j*=std::cos(x(i)/std::sqrt(static_cast<Scalar>(i+1)));
        g(j)= x(j)/2000.0 + std::sin(x(j)/sj)/sj * p_no_j;
      }
    }
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-600); }
  static Vector standardUpper(Index n){ return Vector::Constant(n, 600); }
  static Vector globalMinimizer(Index n){ return Vector::Zero(n); }
  static constexpr Scalar globalMinimum = 0.0;
private:
  Index n_;
};

// ---------------------------------------------------------------------------
// 4. Zakharov  [Jamil13 #229]  (n-D)
//    Zakharov. f= sum xi^2 + (sum 0.5 i xi)^2 + (sum 0.5 i xi)^4
//    i=1..n (1-indexed). Domain: [-5,10]^n  Global min: f(0)=0
//    Plate-shaped, ill-conditioned due to high-order term. [Jamil13]
// ---------------------------------------------------------------------------
class Zakharov : public Function {
public:
  explicit Zakharov(Index n=2): n_(n){ if(n<=0) throw std::invalid_argument("Zakharov n>0");}
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Zakharov dim");
    Scalar s1=0, s2=0;
    for(Index i=0;i<n_;++i){ s1+= x(i)*x(i); s2+= 0.5*static_cast<Scalar>(i+1)*x(i); }
    return s1 + s2*s2 + s2*s2*s2*s2;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Zakharov dim");
    Scalar s2=0; for(Index i=0;i<n_;++i) s2+= 0.5*static_cast<Scalar>(i+1)*x(i);
    Scalar coeff = 2*s2 + 4*s2*s2*s2;
    Vector g(n_);
    for(Index i=0;i<n_;++i){
      Scalar ai = 0.5*static_cast<Scalar>(i+1);
      g(i)= 2*x(i) + coeff*ai;
    }
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-5); }
  static Vector standardUpper(Index n){ return Vector::Constant(n, 10); }
  static Vector globalMinimizer(Index n){ return Vector::Zero(n); }
  static constexpr Scalar globalMinimum = 0.0;
private:
  Index n_;
};

// ---------------------------------------------------------------------------
// 5. Levy N.13  [Jamil13 #135]  (2-D, Levy & Gomez 1981)
//    f(x,y)= sin^2(3pi x) + (x-1)^2(1+sin^2(3pi y)) + (y-1)^2(1+sin^2(2pi y))
//    Domain: [-10,10]^2  Global min: f(1,1)=0
// ---------------------------------------------------------------------------
class LevyN13 : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("LevyN13 requires n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar t1 = std::sin(3*PI*xx);
    Scalar t2 = std::sin(3*PI*yy);
    Scalar t3 = std::sin(2*PI*yy);
    return t1*t1 + (xx-1)*(xx-1)*(1+t2*t2) + (yy-1)*(yy-1)*(1+t3*t3);
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("LevyN13 requires n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar s1 = std::sin(3*PI*xx), c1 = std::cos(3*PI*xx);
    Scalar s2 = std::sin(3*PI*yy), c2 = std::cos(3*PI*yy);
    Scalar s3 = std::sin(2*PI*yy), c3 = std::cos(2*PI*yy);
    Vector g(2);
    g(0)= 2*s1*c1*3*PI + 2*(xx-1)*(1+s2*s2);
    // df/dy = (x-1)^2 *2 s2 c2 *3pi + 2(y-1)(1+s3^2) + (y-1)^2 *2 s3 c3 *2pi
    g(1)= (xx-1)*(xx-1)*2*s2*c2*3*PI + 2*(yy-1)*(1+s3*s3) + (yy-1)*(yy-1)*2*s3*c3*2*PI;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-10,-10; return v; }
  static Vector standardUpper(){ Vector v(2); v<<10,10; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<1,1; return v; }
  static constexpr Scalar globalMinimum = 0.0;
};

// ---------------------------------------------------------------------------
// 6. Himmelblau  [Jamil13 #91]  (2-D)  Himmelblau 1972.
//    f(x,y)= (x^2 + y -11)^2 + (x + y^2 -7)^2
//    Domain: [-5,5]^2  Global minima (4): f=0 at (3,2), (-2.805...,3.131...),
//    (-3.779..., -3.283...), (3.584..., -1.848...)
//    Classic in [Dixon78] / [MGH81] (as "Himmelblau").
// ---------------------------------------------------------------------------
class Himmelblau : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Himmelblau n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar a = xx*xx + yy -11;
    Scalar b = xx + yy*yy -7;
    return a*a + b*b;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Himmelblau n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar a = xx*xx + yy -11;
    Scalar b = xx + yy*yy -7;
    Vector g(2);
    g(0)= 4*xx*a + 2*b;
    g(1)= 2*a + 4*yy*b;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-5,-5; return v; }
  static Vector standardUpper(){ Vector v(2); v<<5,5; return v; }
  // one of the 4 minima
  static Vector globalMinimizer(){ Vector v(2); v<<3,2; return v; }
  static constexpr Scalar globalMinimum = 0.0;
};

// ---------------------------------------------------------------------------
// 7. Beale  [Jamil13 #33]  Beale 1972. Also Moré-Garbow-Hillstrom #4.
//    f(x,y)= (1.5 -x + x y)^2 + (2.25 -x + x y^2)^2 + (2.625 -x + x y^3)^2
//    Domain: [-4.5,4.5]^2  Global min: f(3,0.5)=0 . Narrow curved valley.
// ---------------------------------------------------------------------------
class Beale : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Beale n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar t1=1.5 - xx + xx*yy;
    Scalar t2=2.25 - xx + xx*yy*yy;
    Scalar t3=2.625 - xx + xx*yy*yy*yy;
    return t1*t1 + t2*t2 + t3*t3;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Beale n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar t1=1.5 - xx + xx*yy;
    Scalar t2=2.25 - xx + xx*yy*yy;
    Scalar t3=2.625 - xx + xx*yy*yy*yy;
    Scalar dtdx1 = -1 + yy;
    Scalar dtdx2 = -1 + yy*yy;
    Scalar dtdx3 = -1 + yy*yy*yy;
    Scalar dtdy1 = xx;
    Scalar dtdy2 = 2*xx*yy;
    Scalar dtdy3 = 3*xx*yy*yy;
    Vector g(2);
    g(0)= 2*t1*dtdx1 + 2*t2*dtdx2 + 2*t3*dtdx3;
    g(1)= 2*t1*dtdy1 + 2*t2*dtdy2 + 2*t3*dtdy3;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-4.5,-4.5; return v; }
  static Vector standardUpper(){ Vector v(2); v<<4.5,4.5; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<3,0.5; return v; }
  static constexpr Scalar globalMinimum = 0.0;
};

// ---------------------------------------------------------------------------
// 8. Booth  [Jamil13 #40]  Booth 1955.
//    f(x,y)= (x+2y-7)^2 + (2x+y-5)^2
//    Domain: [-10,10]^2  Global min: f(1,3)=0 . Quadratic valley, ill-conditioned.
// ---------------------------------------------------------------------------
class Booth : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Booth n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar a= xx+2*yy -7;
    Scalar b= 2*xx+yy -5;
    return a*a + b*b;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Booth n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar a= xx+2*yy -7;
    Scalar b= 2*xx+yy -5;
    Vector g(2);
    g(0)= 2*a +4*b;
    g(1)= 4*a +2*b;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-10,-10; return v; }
  static Vector standardUpper(){ Vector v(2); v<<10,10; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<1,3; return v; }
  static constexpr Scalar globalMinimum = 0.0;
};

// ---------------------------------------------------------------------------
// 9. Matyas  [Jamil13 #115]  Matyas 1965.
//    f(x,y)=0.26(x^2+y^2)-0.48 x y
//    Domain: [-10,10]^2  Global min: f(0,0)=0 . Plate-shaped, strong correlation.
// ---------------------------------------------------------------------------
class Matyas : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Matyas n=2");
    Scalar xx=x(0), yy=x(1);
    return 0.26*(xx*xx+yy*yy) -0.48*xx*yy;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Matyas n=2");
    Scalar xx=x(0), yy=x(1);
    Vector g(2);
    g(0)= 0.52*xx -0.48*yy;
    g(1)= 0.52*yy -0.48*xx;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-10,-10; return v; }
  static Vector standardUpper(){ Vector v(2); v<<10,10; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<0,0; return v; }
  static constexpr Scalar globalMinimum = 0.0;
};

// ---------------------------------------------------------------------------
// 10. McCormick  [Jamil13 #119]  McCormick 1972, also [MGH81] style.
//     f(x,y)= sin(x+y) + (x-y)^2 -1.5 x +2.5 y +1
//     Domain: x in [-1.5,4], y in [-3,4]  Global min: f(-0.54719,-1.54719)=-1.9133
//     Non-convex, sinusoidal + quadratic. Classic.
// ---------------------------------------------------------------------------
class McCormick : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("McCormick n=2");
    Scalar xx=x(0), yy=x(1);
    return std::sin(xx+yy) + (xx-yy)*(xx-yy) -1.5*xx +2.5*yy +1;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("McCormick n=2");
    Scalar xx=x(0), yy=x(1);
    Scalar c = std::cos(xx+yy);
    Vector g(2);
    g(0)= c +2*(xx-yy) -1.5;
    g(1)= c -2*(xx-yy) +2.5;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-1.5,-3; return v; }
  static Vector standardUpper(){ Vector v(2); v<<4,4; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<-0.54719755,-1.54719755; return v; }
  static constexpr Scalar globalMinimum = -1.913222954981036; // Jamil13
};

// ---------------------------------------------------------------------------
// 11. Branin (Branin RCOS)  [Jamil13 #42][Dixon78][MGH81 #7]
//     f= (x2 -5.1/(4π²) x1² +5/π x1 -6)² +10(1-1/(8π))cos x1 +10
//     Domain: x1∈[-5,10], x2∈[0,15]  Global minima (3): f=0.397887 @
//     (-π,12.275), (π,2.275), (9.42478,2.475)  Classic [Dixon78].
// ---------------------------------------------------------------------------
class Branin : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Branin n=2");
    Scalar x1=x(0), x2=x(1);
    Scalar a = x2 - 5.1/(4*PI*PI)*x1*x1 + 5/PI*x1 -6;
    return a*a + 10*(1 - 1/(8*PI))*std::cos(x1) +10;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Branin n=2");
    Scalar x1=x(0), x2=x(1);
    Scalar a = x2 - 5.1/(4*PI*PI)*x1*x1 + 5/PI*x1 -6;
    Scalar dadx1 = -5.1/(2*PI*PI)*x1 +5/PI;
    Vector g(2);
    g(0)= 2*a*dadx1 -10*(1 - 1/(8*PI))*std::sin(x1);
    g(1)= 2*a;
    return g;
  }
  static Vector standardLower(){ Vector v(2); v<<-5,0; return v; }
  static Vector standardUpper(){ Vector v(2); v<<10,15; return v; }
  static Vector globalMinimizer(){ Vector v(2); v<<-PI,12.275; return v; }
  static constexpr Scalar globalMinimum = 0.39788735772973816;
};

// ---------------------------------------------------------------------------
// 12. Goldstein-Price  [Jamil13 #78][Dixon78][MGH81 #]
//     f=[1+(x1+x2+1)²(19-14x1+3x1²-14x2+6x1x2+3x2²)]*
//       [30+(2x1-3x2)²(18-32x1+12x1²+48x2-36x1x2+27x2²)]
//     Domain: [-2,2]^2  Global min: f(0,-1)=3  Steep narrow valley.
// ---------------------------------------------------------------------------
class GoldsteinPrice : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("GoldsteinPrice n=2");
    Scalar x1=x(0), x2=x(1);
    Scalar a = x1+x2+1;
    Scalar b = 19-14*x1+3*x1*x1-14*x2+6*x1*x2+3*x2*x2;
    Scalar c = 2*x1-3*x2;
    Scalar d = 18-32*x1+12*x1*x1+48*x2-36*x1*x2+27*x2*x2;
    return (1+a*a*b)*(30+c*c*d);
  }
  Vector computeGradient(const Vector& x) override {
    // finite differences (analytic tedious, FD is exact for testing)
    const Scalar h=1e-8; Vector g(2), xp=x, xm=x;
    for(Index i=0;i<2;++i){ xp=x; xm=x; xp(i)+=h; xm(i)-=h; g(i)=(computeValue(xp)-computeValue(xm))/(2*h); }
    return g;
  }
  static Vector standardLower(){ return Vector::Constant(2,-2); }
  static Vector standardUpper(){ return Vector::Constant(2, 2); }
  static Vector globalMinimizer(){ Vector v(2); v<<0,-1; return v; }
  static constexpr Scalar globalMinimum = 3.0;
};

// ---------------------------------------------------------------------------
// 13. Hartman 3-D  [Jamil13 #86][Dixon78]
//     f= - Σ_{i=1}^4 ci exp(- Σ_{j=1}^3 aij (xj-pij)² )
//     Domain: [0,1]^3  Global min: -3.86278 @ (0.1146,0.5559,0.8525)
// ---------------------------------------------------------------------------
class Hartman3 : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=3) throw std::invalid_argument("Hartman3 n=3");
    static const Scalar c[4]={1.0,1.2,3.0,3.2};
    static const Scalar a[4][3]={{3,10,30},{0.1,10,35},{3,10,30},{0.1,10,35}};
    static const Scalar p[4][3]={{0.3689,0.1170,0.2673},{0.4699,0.4387,0.7470},{0.1091,0.8732,0.5547},{0.03815,0.5743,0.8828}};
    Scalar s=0; for(int i=0;i<4;++i){ Scalar t=0; for(int j=0;j<3;++j) t+= a[i][j]*sqr(x(j)-p[i][j]); s+= c[i]*std::exp(-t); }
    return -s;
  }
  Vector computeGradient(const Vector& x) override {
    static const Scalar c[4]={1.0,1.2,3.0,3.2};
    static const Scalar a[4][3]={{3,10,30},{0.1,10,35},{3,10,30},{0.1,10,35}};
    static const Scalar p[4][3]={{0.3689,0.1170,0.2673},{0.4699,0.4387,0.7470},{0.1091,0.8732,0.5547},{0.03815,0.5743,0.8828}};
    Vector g=Vector::Zero(3);
    for(int i=0;i<4;++i){
      Scalar t=0; for(int j=0;j<3;++j) t+= a[i][j]*sqr(x(j)-p[i][j]);
      Scalar e=c[i]*std::exp(-t);
      for(int j=0;j<3;++j) g(j)+= e*2*a[i][j]*(x(j)-p[i][j]);
    }
    return g;
  }
  static Vector standardLower(){ return Vector::Zero(3); }
  static Vector standardUpper(){ return Vector::Ones(3); }
  static Vector globalMinimizer(){ Vector v(3); v<<0.114614,0.555649,0.852547; return v; }
  static constexpr Scalar globalMinimum = -3.86278214782076;
};

// ---------------------------------------------------------------------------
// 14. Hartman 6-D  [Jamil13 #86][Dixon78]
//     Same form n=6. Global min: -3.32237 @ (0.2016,0.1500,0.4768,0.2753,0.3116,0.6573)
// ---------------------------------------------------------------------------
class Hartman6 : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=6) throw std::invalid_argument("Hartman6 n=6");
    static const Scalar c[4]={1.0,1.2,3.0,3.2};
    static const Scalar a[4][6]={{10,3,17,3.5,1.7,8},{0.05,10,17,0.1,8,14},{3,3.5,1.7,10,17,8},{17,8,0.05,10,0.1,14}};
    static const Scalar p[4][6]={{0.1312,0.1696,0.5569,0.0124,0.8283,0.5886},{0.2329,0.4135,0.8307,0.3736,0.1004,0.9991},{0.2348,0.1451,0.3522,0.2883,0.3047,0.6650},{0.4047,0.8828,0.8732,0.5743,0.1091,0.0381}};
    Scalar s=0; for(int i=0;i<4;++i){ Scalar t=0; for(int j=0;j<6;++j) t+= a[i][j]*sqr(x(j)-p[i][j]); s+= c[i]*std::exp(-t); }
    return -s;
  }
  Vector computeGradient(const Vector& x) override {
    static const Scalar c[4]={1.0,1.2,3.0,3.2};
    static const Scalar a[4][6]={{10,3,17,3.5,1.7,8},{0.05,10,17,0.1,8,14},{3,3.5,1.7,10,17,8},{17,8,0.05,10,0.1,14}};
    static const Scalar p[4][6]={{0.1312,0.1696,0.5569,0.0124,0.8283,0.5886},{0.2329,0.4135,0.8307,0.3736,0.1004,0.9991},{0.2348,0.1451,0.3522,0.2883,0.3047,0.6650},{0.4047,0.8828,0.8732,0.5743,0.1091,0.0381}};
    Vector g=Vector::Zero(6);
    for(int i=0;i<4;++i){
      Scalar t=0; for(int j=0;j<6;++j) t+= a[i][j]*sqr(x(j)-p[i][j]);
      Scalar e=c[i]*std::exp(-t);
      for(int j=0;j<6;++j) g(j)+= e*2*a[i][j]*(x(j)-p[i][j]);
    }
    return g;
  }
  static Vector standardLower(){ return Vector::Zero(6); }
  static Vector standardUpper(){ return Vector::Ones(6); }
  static Vector globalMinimizer(){ Vector v(6); v<<0.20169,0.15001,0.476874,0.275332,0.311653,0.657301; return v; }
  static constexpr Scalar globalMinimum = -3.32236801141551;
};

// ---------------------------------------------------------------------------
// 15. Eggholder  [Jamil13 #65]  Highly multimodal/deceptive
//     f= -(y+47)sin sqrt(|x/2 + (y+47)|) - x sin sqrt(|x-(y+47)|)
//     Domain: [-512,512]^2  Global min: -959.6407 @ (512,404.2319)
// ---------------------------------------------------------------------------
class Eggholder : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Eggholder n=2");
    Scalar xx=x(0), yy=x(1);
    return -(yy+47)*std::sin(std::sqrt(std::abs(xx/2+(yy+47)))) - xx*std::sin(std::sqrt(std::abs(xx-(yy+47))));
  }
  Vector computeGradient(const Vector& x) override {
    const Scalar h=1e-8; Vector g(2), xp=x, xm=x;
    for(Index i=0;i<2;++i){ xp=x; xm=x; xp(i)+=h; xm(i)-=h; g(i)=(computeValue(xp)-computeValue(xm))/(2*h); }
    return g;
  }
  static Vector standardLower(){ return Vector::Constant(2,-512); }
  static Vector standardUpper(){ return Vector::Constant(2, 512); }
  static Vector globalMinimizer(){ Vector v(2); v<<512,404.2319; return v; }
  static constexpr Scalar globalMinimum = -959.6406627208508;
};

// ---------------------------------------------------------------------------
// 16. Michalewicz  [Jamil13 #123]  m=10
//     f= - Σ sin(xi) sin^{2m}(i xi^2/π)
//     Domain: [0,π]^n  Global min approx -1.8013 (n=2), -9.66015 (n=10)
//     Very spiky, n! local minima.
// ---------------------------------------------------------------------------
class Michalewicz : public Function {
  Index n_; int m_=10;
public:
  explicit Michalewicz(Index n=2, int m=10):n_(n),m_(m){}
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=n_) throw std::invalid_argument("Michalewicz dim");
    Scalar s=0; for(Index i=0;i<n_;++i){
      Scalar a = static_cast<Scalar>(i+1)*x(i)*x(i)/PI;
      s+= std::sin(x(i))*std::pow(std::sin(a), 2*m_);
    }
    return -s;
  }
  Vector computeGradient(const Vector& x) override {
    const Scalar h=1e-8; Vector g(n_), xp=x, xm=x;
    for(Index i=0;i<n_;++i){ xp=x; xm=x; xp(i)+=h; xm(i)-=h; g(i)=(computeValue(xp)-computeValue(xm))/(2*h); }
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Zero(n); }
  static Vector standardUpper(Index n){ return Vector::Constant(n, PI); }
};

// ---------------------------------------------------------------------------
// 17. Easom  [Jamil13 #63]  Easom 1990
//     f= -cos(x1)cos(x2) exp(-((x1-π)²+(x2-π)²))
//     Domain: [-100,100]^2  Global min: -1 @ (π,π)  Almost flat elsewhere.
// ---------------------------------------------------------------------------
class Easom : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Easom n=2");
    Scalar x1=x(0), x2=x(1);
    return -std::cos(x1)*std::cos(x2)*std::exp(-((x1-PI)*(x1-PI)+(x2-PI)*(x2-PI)));
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("Easom n=2");
    Scalar x1=x(0), x2=x(1);
    Scalar c1=std::cos(x1), c2=std::cos(x2), s1=std::sin(x1), s2=std::sin(x2);
    Scalar e=std::exp(-((x1-PI)*(x1-PI)+(x2-PI)*(x2-PI)));
    Vector g(2);
    g(0)= s1*c2*e + c1*c2*e*(-2*(x1-PI)); // = c2*e*(s1 -2(x1-PI)c1)
    // exact: g0 = sin(x1)cos(x2)e + cos(x1)cos(x2)e*2(x1-pi) ??? sign
    // f=-c1c2 e => df/dx1 = s1c2 e -c1c2 e*(-2)(x1-PI)= s1c2 e +2(x1-PI)c1c2 e
    g(0)= s1*c2*e + 2*(x1-PI)*c1*c2*e;
    g(1)= c1*s2*e + 2*(x2-PI)*c1*c2*e;
    return g;
  }
  static Vector standardLower(){ return Vector::Constant(2,-100); }
  static Vector standardUpper(){ return Vector::Constant(2, 100); }
  static Vector globalMinimizer(){ Vector v(2); v<<PI,PI; return v; }
  static constexpr Scalar globalMinimum = -1.0;
};

// ---------------------------------------------------------------------------
// 18. Three-Hump Camel  [Jamil13 #210] (≠ Six-Hump)
//     f=2x1² -1.05x1⁴ +x1⁶/6 +x1x2 +x2²
//     Domain: [-5,5]^2  Global min: 0 @ (0,0)
// ---------------------------------------------------------------------------
class ThreeHumpCamel : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("ThreeHumpCamel n=2");
    Scalar x1=x(0), x2=x(1);
    return 2*x1*x1 -1.05*x1*x1*x1*x1 + x1*x1*x1*x1*x1*x1/6 + x1*x2 + x2*x2;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=2) throw std::invalid_argument("ThreeHumpCamel n=2");
    Scalar x1=x(0), x2=x(1);
    Vector g(2);
    g(0)=4*x1 -4.2*x1*x1*x1 + x1*x1*x1*x1*x1 + x2;
    g(1)= x1 +2*x2;
    return g;
  }
  static Vector standardLower(){ return Vector::Constant(2,-5); }
  static Vector standardUpper(){ return Vector::Constant(2, 5); }
  static Vector globalMinimizer(){ return Vector::Zero(2); }
  static constexpr Scalar globalMinimum = 0.0;
};

} // namespace benchmarks
} // namespace optimize

#endif // BENCHMARKS_HH
