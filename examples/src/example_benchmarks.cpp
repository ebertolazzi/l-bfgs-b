// Copyright (c) 2026 Enrico Bertolazzi
// Distributed under the terms of the MIT License
//
// Comprehensive hard benchmark suite for L-BFGS-B / L-BFGS
// Sources:
//   [Jamil13] M. Jamil & X.-S. Yang, "A literature survey of benchmark
//             functions for global optimisation", IJMMNO 4(2), 150-194 (2013).
//   [Dixon78] L.C.W. Dixon & G.P. Szegö (eds.), Towards Global Optimisation 2,
//             North-Holland, 1978.
//   [MGH81]   J.J. Moré, B.S. Garbow & K.E. Hillstrom, "Testing unconstrained
//             optimization software", ACM TOMS 7(1), 17-41 (1981).
//   Test suite legacy from l-bfgs-b/test/include/test.h (Forrester, Simple,
//   NonSmooth2D, Rosenbrock, SixHumpCamel, Spiral).
//
// Ogni funzione è lanciata sia unconstrained che box-constrained (dove
// previsto) e i risultati sono riassunti in tabella unicode finale.

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <cstring>
#include <random>
#include "lbfgsb.hh"
#include "benchmarks.hh"

using namespace optimize;
using namespace optimize::benchmarks;

// Verbose callback to print detailed iteration info
static bool g_verbose = false;

void verboseCallback(Solver* solver) {
  if (!g_verbose) return;
  const State& st = solver->state();
  std::cout << "  iter=" << std::setw(4) << st.iter()
            << " feval=" << std::setw(5) << st.fEvals()
            << " geval=" << std::setw(5) << st.gEvals()
            << " f=" << std::fixed << std::setprecision(6) << std::setw(12) << st.f()
            << " |g|=" << std::scientific << std::setprecision(2) << std::setw(10) << st.gNorm()
            << " df=" << std::scientific << std::setprecision(2) << std::setw(10) << st.dfNorm()
            << " dx=" << std::scientific << std::setprecision(2) << std::setw(10) << st.dxNorm()
            << "\n";
}

// ---------------------------------------------------------------------------
// Funzioni legacy difficili già presenti in test/include/test.h e
// examples/include/example.h – copiate qui per avere un unico eseguibile
// di benchmark.  Manteniamo i nomi originali per tracciabilità.
// ---------------------------------------------------------------------------

// Forrester [Jamil13 – Forrester et al. 2008, http://www.sfu.ca/~ssurjano/forretal08.html]
class ForresterLegacy : public Function {
public:
  Scalar computeValue(const Vector& x) override { return (6*x(0)-2)*(6*x(0)-2)*std::sin(12*x(0)-4); }
  Vector computeGradient(const Vector& x) override {
    return Vector{{ 12*(6*x(0)-2)*( std::sin(12*x(0)-4) + (6*x(0)-2)*std::cos(12*x(0)-4)) }};
  }
};

// Simple – Cornell line-search example
class SimpleLegacy : public Function {
public:
  Scalar computeValue(const Vector& x) override { return x(0)-x(1)+2*x(0)*x(1)+2*x(0)*x(0)+x(1)*x(1); }
  Vector computeGradient(const Vector& x) override { return Vector{{1+2*x(1)+4*x(0), -1+2*x(0)+2*x(1)}}; }
};

// NonSmooth2D – |x^{2/3} y^{2/3}+...|  non-smooth al minimo [Jamil13 – non-smooth]
class NonSmooth2DLegacy : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    return std::abs(std::cbrt(x(0)*x(0))*std::cbrt(x(1)*x(1)) + std::cbrt(x(0)*x(0)) + std::cbrt(x(1)*x(1)));
  }
  Vector computeGradient(const Vector& x) override {
    Vector g(x.size());
    Scalar fx = computeValue(x);
    Scalar s = fx>0?1:fx<0?-1:0;
    g(0)= x(0)!=0? 2.0*(std::cbrt(x(1)*x(1))+1)*s/(3.0*std::cbrt(x(0))) : ScalarLimits::max();
    g(1)= x(1)!=0? 2.0*(std::cbrt(x(0)*x(0))+1)*s/(3.0*std::cbrt(x(1))) : ScalarLimits::max();
    return g;
  }
};

// Six-Hump Camel [Jamil13 #...][MGH81][Dixon78]
class SixHumpCamelLegacy : public Function {
public:
  Scalar computeValue(const Vector& x) override { return x(0)*x(0)*(4-2.1*x(0)*x(0)+x(0)*x(0)*x(0)*x(0)/3)+x(0)*x(1)+4*x(1)*x(1)*(-1+x(1)*x(1)); }
  Vector computeGradient(const Vector& x) override { return Vector{{8*x(0)-8.4*x(0)*x(0)*x(0)+2*x(0)*x(0)*x(0)*x(0)*x(0)+x(1), x(0)-8*x(1)+16*x(1)*x(1)*x(1)}}; }
};

// Spirale – path planning con energia di bending + pose (vedi test.h Spiral)
struct PoseLL { double x,y,th; PoseLL(double xx=0,double yy=0,double tth=0):x(xx),y(yy),th(tth){} };
class SpiralLegacy : public Function {
  PoseLL p0{0,0,0}, pf{5,5,M_PI/3}, w{25,25,30};
public:
  Scalar computeValue(const Vector& x) override {
    Scalar p1=x(0),p2=x(1),p4=x(2);
    Scalar be=27*p4*(4*p1*p1 -p1*p2+4*p2*p2)/280;
    auto cth=[&](Scalar a){return std::cos(a);}; auto sth=[&](Scalar a){return std::sin(a);};
    Scalar xe_t1=(p4*(cth(p0.th)+4*cth(0.056488037109375*p1*p4-0.024261474609375*p2*p4+p0.th)+2*cth(0.17724609375*p1*p4-0.06005859375*p2*p4+p0.th)+4*cth(0.304046630859375*p1*p4-0.066741943359375*p2*p4+p0.th)+cth(0.375*p1*p4+0.375*p2*p4+p0.th)+2*cth(0.3984375*p1*p4-0.0234375*p2*p4+p0.th)+4*cth(0.399261474609375*p1*p4+0.318511962890625*p2*p4+p0.th)+2*cth(0.43505859375*p1*p4+0.19775390625*p2*p4+p0.th)+4*cth(0.441741943359375*p1*p4+0.070953369140625*p2*p4+p0.th))-24*pf.x+24*p0.x);
    Scalar xe=xe_t1*xe_t1/576; Scalar ye_t1=(p4*(sth(p0.th)+4*sth(0.056488037109375*p1*p4-0.024261474609375*p2*p4+p0.th)+2*sth(0.17724609375*p1*p4-0.06005859375*p2*p4+p0.th)+4*sth(0.304046630859375*p1*p4-0.066741943359375*p2*p4+p0.th)+sth(0.375*p1*p4+0.375*p2*p4+p0.th)+2*sth(0.3984375*p1*p4-0.0234375*p2*p4+p0.th)+4*sth(0.399261474609375*p1*p4+0.318511962890625*p2*p4+p0.th)+2*sth(0.43505859375*p1*p4+0.19775390625*p2*p4+p0.th)+4*sth(0.441741943359375*p1*p4+0.070953369140625*p2*p4+p0.th))-24*pf.y+24*p0.y);
    Scalar ye=ye_t1*ye_t1/576; Scalar the_t1=0.375*p1*p4+0.375*p2*p4-pf.th+p0.th; Scalar the=the_t1*the_t1;
    return be+w.x*xe+w.y*ye+w.th*the;
  }
  Vector computeGradient(const Vector& x) override {
    // differenze finite centrali per brevità (funzione molto lunga analitica in test.h)
    const Scalar h=1e-7; Vector g(x.size()), xp=x, xm=x;
    for(Index i=0;i<x.size();++i){ xp=x; xm=x; xp(i)+=h; xm(i)-=h; g(i)=(computeValue(xp)-computeValue(xm))/(2*h); }
    return g;
  }
};

// Rosenbrock n-D – già in benchmarks? No, usiamo quello classico di test.h
class RosenbrockLegacy : public Function {
  static constexpr Scalar b=100;
public:
  Scalar computeValue(const Vector& x) override {
    Scalar v=0; for(Index i=0;i<x.size()-1;++i){ Scalar t1=x(i+1)-x(i)*x(i); Scalar t2=x(i)-1; v+= b*t1*t1 + t2*t2; } return v;
  }
  Vector computeGradient(const Vector& x) override {
    Vector g(x.size()); g.setZero();
    for(Index i=0;i<x.size()-1;++i){
      if(i==0) g(i)=4*b*(x(i)*x(i)*x(i)-x(i)*x(i+1))+2*x(0)-2;
      if(i>0 && i<x.size()-1) g(i)=4*b*(x(i)*x(i)*x(i)-x(i)*x(i+1))+2*b*(x(i)-x(i-1)*x(i-1))+2*x(i)-2;
      if(i+1==x.size()-1) g(i+1)=2*b*(x(i+1)-x(i)*x(i));
    }
    return g;
  }
};

// ---------------------------------------------------------------------------
// Extra hard benchmarks da Jamil13 non inclusi nei 10 base (per arricchire)
// ---------------------------------------------------------------------------

// Schwefel  [Jamil13 #196]  f = 418.9829*n - sum xi sin sqrt(|xi|)  Domain [-500,500]^n  min 0 @420.9687
class Schwefel : public Function {
  Index n_;
public:
  explicit Schwefel(Index n=2):n_(n){}
  Scalar computeValue(const Vector& x) override {
    Scalar s=0; for(Index i=0;i<n_;++i) s+= x(i)*std::sin(std::sqrt(std::abs(x(i))));
    return 418.9829*static_cast<Scalar>(n_) - s;
  }
  Vector computeGradient(const Vector& x) override {
    Vector g(n_); for(Index i=0;i<n_;++i){
      Scalar xi=x(i); Scalar ax=std::abs(xi); if(ax<1e-12){g(i)=0; continue;}
      Scalar s=std::sin(std::sqrt(ax)), c=std::cos(std::sqrt(ax));
      g(i)= -( s + xi*c/(2*std::sqrt(ax)) * (xi>=0?1:-1) );
    } return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-500); }
  static Vector standardUpper(Index n){ return Vector::Constant(n, 500); }
  static Vector globalMinimizer(Index n){ return Vector::Constant(n,420.968746); }
  static constexpr Scalar globalMinimum = 0.0;
};

// Dixon-Price  [Jamil13 #60]  f = (x1-1)^2 + sum i*(2 xi^2 - x_{i-1})^2
class DixonPrice : public Function {
  Index n_;
public:
  explicit DixonPrice(Index n=4):n_(n){}
  Scalar computeValue(const Vector& x) override {
    Scalar v=(x(0)-1)*(x(0)-1);
    for(Index i=1;i<n_;++i){ Scalar t=2*x(i)*x(i)-x(i-1); v+= static_cast<Scalar>(i+1)*t*t; }
    return v;
  }
  Vector computeGradient(const Vector& x) override {
    Vector g=Vector::Zero(n_);
    g(0)=2*(x(0)-1);
    for(Index i=1;i<n_;++i){
      Scalar t=2*x(i)*x(i)-x(i-1);
      g(i-1) += static_cast<Scalar>(i+1)*2*t*(-1);
      g(i)   += static_cast<Scalar>(i+1)*2*t*4*x(i);
    }
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-10); }
  static Vector standardUpper(Index n){ return Vector::Constant(n,10); }
};

// Styblinski-Tang  [Jamil13 #202]  f = 0.5 sum(xi^4 -16 xi^2 +5 xi)  Domain [-5,5]^n  min -39.16599*n @ -2.903534
class StyblinskiTang : public Function {
  Index n_;
public:
  explicit StyblinskiTang(Index n=2):n_(n){}
  Scalar computeValue(const Vector& x) override {
    Scalar s=0; for(Index i=0;i<n_;++i) s+= x(i)*x(i)*x(i)*x(i) -16*x(i)*x(i) +5*x(i);
    return 0.5*s;
  }
  Vector computeGradient(const Vector& x) override {
    Vector g(n_); for(Index i=0;i<n_;++i) g(i)=0.5*(4*x(i)*x(i)*x(i)-32*x(i)+5);
    return g;
  }
  static Vector standardLower(Index n){ return Vector::Constant(n,-5); }
  static Vector standardUpper(Index n){ return Vector::Constant(n,5); }
  static Vector globalMinimizer(Index n){ return Vector::Constant(n,-2.903534); }
};

// Powell singular [MGH81 #??] – 4-D Powell
class Powell : public Function {
public:
  Scalar computeValue(const Vector& x) override {
    // f = (x1+10x2)^2 +5(x3-x4)^2 + (x2-2x3)^4 +10(x1-x4)^4   x in R^4
    if(x.size()!=4) throw std::invalid_argument("Powell n=4");
    Scalar a=x(0)+10*x(1), b=x(2)-x(3), c=x(1)-2*x(2), d=x(0)-x(3);
    return a*a +5*b*b + c*c*c*c +10*d*d*d*d;
  }
  Vector computeGradient(const Vector& x) override {
    if(x.size()!=4) throw std::invalid_argument("Powell n=4");
    Scalar a=x(0)+10*x(1), b=x(2)-x(3), c=x(1)-2*x(2), d=x(0)-x(3);
    Vector g(4);
    g(0)=2*a +40*d*d*d;
    g(1)=20*a +4*c*c*c;
    g(2)=10*b -8*c*c*c;
    g(3)=-10*b -40*d*d*d;
    return g;
  }
};

// ---------------------------------------------------------------------------

struct BenchCase {
  std::string name;
  std::function<Function*()> factory;
  Vector x0;
  Vector l;
  Vector u;
  Vector x_star;
  Scalar f_star;
};

struct BenchResult {
  std::string name;
  int n;
  std::string bounds;
  int iter_min;
  int iter_max;
  double iter_avg;
  int ok_count;
  int fail_count;
  Scalar f_min;
  Scalar f_max;
  Scalar g_min;
  Scalar g_max;
  Scalar f_star;
  Scalar err_f;
  Scalar g_norm;
  bool success;
  std::string status;
};

int main(int argc, char* argv[]) {
  // RNG C++17
  std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  // Parse --verbose flag
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--verbose") == 0) {
      g_verbose = true;
    }
  }

  std::vector<BenchCase> cases;

  // 10 benchmark base (unconst+box)
  auto mk = [&](std::string base, std::function<Function*()> fac, Vector x0, Vector l,Vector u, Vector xs,Scalar fs){
    cases.push_back({base+" (free)", fac, x0, Vector(), Vector(), xs, fs});
    cases.push_back({base+" (box)", fac, x0, l, u, xs, fs});
  };

  // scalabili n=5
  mk("Ackley n=5", [](){return new Ackley(5);}, Vector::Constant(5,1.0), Ackley::standardLower(5), Ackley::standardUpper(5), Ackley::globalMinimizer(5), 0.0);
  mk("Rastrigin n=5", [](){return new Rastrigin(5);}, Vector::Constant(5,1.0), Rastrigin::standardLower(5), Rastrigin::standardUpper(5), Rastrigin::globalMinimizer(5),0.0);
  mk("Griewank n=5", [](){return new Griewank(5);}, Vector::Constant(5,2.0), Griewank::standardLower(5), Griewank::standardUpper(5), Griewank::globalMinimizer(5),0.0);
  mk("Zakharov n=5", [](){return new Zakharov(5);}, Vector::Constant(5,1.0), Zakharov::standardLower(5), Zakharov::standardUpper(5), Zakharov::globalMinimizer(5),0.0);
  mk("Schwefel n=5", [](){return new Schwefel(5);}, Vector::Constant(5,400), Schwefel::standardLower(5), Schwefel::standardUpper(5), Schwefel::globalMinimizer(5),0.0);
  mk("DixonPrice n=5", [](){return new DixonPrice(5);}, Vector::Constant(5,-1.0), DixonPrice::standardLower(5), DixonPrice::standardUpper(5), Vector::Zero(5),0.0);
  mk("StyblinskiTang n=5", [](){return new StyblinskiTang(5);}, Vector::Constant(5,0.0), StyblinskiTang::standardLower(5), StyblinskiTang::standardUpper(5), StyblinskiTang::globalMinimizer(5), -39.16599*5);
  // scalabili n=10
  mk("Ackley n=10", [](){return new Ackley(10);}, Vector::Constant(10,1.0), Ackley::standardLower(10), Ackley::standardUpper(10), Ackley::globalMinimizer(10), 0.0);
  mk("Rastrigin n=10", [](){return new Rastrigin(10);}, Vector::Constant(10,1.0), Rastrigin::standardLower(10), Rastrigin::standardUpper(10), Rastrigin::globalMinimizer(10),0.0);
  mk("Griewank n=10", [](){return new Griewank(10);}, Vector::Constant(10,2.0), Griewank::standardLower(10), Griewank::standardUpper(10), Griewank::globalMinimizer(10),0.0);
  mk("Zakharov n=10", [](){return new Zakharov(10);}, Vector::Constant(10,1.0), Zakharov::standardLower(10), Zakharov::standardUpper(10), Zakharov::globalMinimizer(10),0.0);
  mk("Schwefel n=10", [](){return new Schwefel(10);}, Vector::Constant(10,400), Schwefel::standardLower(10), Schwefel::standardUpper(10), Schwefel::globalMinimizer(10),0.0);
  mk("DixonPrice n=10", [](){return new DixonPrice(10);}, Vector::Constant(10,-1.0), DixonPrice::standardLower(10), DixonPrice::standardUpper(10), Vector::Zero(10),0.0);
  mk("StyblinskiTang n=10", [](){return new StyblinskiTang(10);}, Vector::Constant(10,0.0), StyblinskiTang::standardLower(10), StyblinskiTang::standardUpper(10), StyblinskiTang::globalMinimizer(10), -39.16599*10);
  mk("Michalewicz n=10", [](){return new Michalewicz(10);}, Vector::Constant(10,1.5), Michalewicz::standardLower(10), Michalewicz::standardUpper(10), Vector::Constant(10,2.20319), -9.66015); // best known
  mk("Michalewicz n=15", [](){return new Michalewicz(15);}, Vector::Constant(15,1.5), Michalewicz::standardLower(15), Michalewicz::standardUpper(15), Vector::Constant(15,2.20319), -14.49022); // best known
  mk("Michalewicz n=20", [](){return new Michalewicz(20);}, Vector::Constant(20,1.5), Michalewicz::standardLower(20), Michalewicz::standardUpper(20), Vector::Constant(20,2.20319), -19.32030); // best known
  // 2-D
  mk("LevyN13", [](){return new LevyN13();}, Vector{{2,2}}, LevyN13::standardLower(), LevyN13::standardUpper(), LevyN13::globalMinimizer(),0.0);
  mk("Himmelblau", [](){return new Himmelblau();}, Vector{{0,0}}, Himmelblau::standardLower(), Himmelblau::standardUpper(), Himmelblau::globalMinimizer(),0.0);
  mk("Beale", [](){return new Beale();}, Vector{{0,0}}, Beale::standardLower(), Beale::standardUpper(), Beale::globalMinimizer(),0.0);
  mk("Booth", [](){return new Booth();}, Vector{{0,0}}, Booth::standardLower(), Booth::standardUpper(), Booth::globalMinimizer(),0.0);
  mk("Matyas", [](){return new Matyas();}, Vector{{5,5}}, Matyas::standardLower(), Matyas::standardUpper(), Matyas::globalMinimizer(),0.0);
  mk("McCormick", [](){return new McCormick();}, Vector{{0,0}}, McCormick::standardLower(), McCormick::standardUpper(), McCormick::globalMinimizer(), McCormick::globalMinimum);
  // legacy difficili da test.h
  cases.push_back({"Rosenbrock n=3 (free)", [](){return new RosenbrockLegacy();}, Vector{{8,-5,3}}, Vector(), Vector(), Vector::Ones(3),0.0});
  cases.push_back({"Rosenbrock n=3 (box 1 act)", [](){return new RosenbrockLegacy();}, Vector{{8,-5,3}}, Vector::Constant(3,-10), Vector{{0.5,10,10}}, Vector{{0.5,0.25,0.0625}},0.0}); // approx
  cases.push_back({"SixHumpCamel (free)", [](){return new SixHumpCamelLegacy();}, Vector{{1,0}}, Vector::Constant(2,-2), Vector::Constant(2,2), Vector{{0.089842,-0.712656}}, -1.031628});
  cases.push_back({"Simple 2D (free)", [](){return new SimpleLegacy();}, Vector{{9,-8}}, Vector::Constant(2,-10), Vector::Constant(2,10), Vector{{-0.5,-0.5}}, -1.25});
  cases.push_back({"NonSmooth2D (free)", [](){return new NonSmooth2DLegacy();}, Vector{{-9,8}}, Vector::Constant(2,-10), Vector::Constant(2,10), Vector::Zero(2),0.0});
  cases.push_back({"Spiral 3D (free)", [](){return new SpiralLegacy();}, Vector{{0,0,7.07}}, Vector(), Vector(), Vector{{0,0,0}},0.0}); // target ~0.3 con vincoli
  cases.push_back({"Forrester 1D (box)", [](){return new ForresterLegacy();}, Vector{{0.5241}}, Vector{{0}}, Vector{{1}}, Vector{{0.25}}, -6.02074});
  cases.push_back({"Powell 4D (free)", [](){return new Powell();}, Vector{{3,-1,0,1}}, Vector(), Vector(), Vector::Zero(4),0.0});
  cases.push_back({"Powell 4D (box)", [](){return new Powell();}, Vector{{3,-1,0,1}}, Vector::Constant(4,-5), Vector::Constant(4,5), Vector::Zero(4),0.0});
  // --- nuovi difficili da Jamil13 ---
  mk("Branin", [](){return new Branin();}, Vector{{0,0}}, Branin::standardLower(), Branin::standardUpper(), Branin::globalMinimizer(), Branin::globalMinimum);
  mk("GoldsteinPrice", [](){return new GoldsteinPrice();}, Vector{{0.5,0.25}}, GoldsteinPrice::standardLower(), GoldsteinPrice::standardUpper(), GoldsteinPrice::globalMinimizer(), GoldsteinPrice::globalMinimum);
  mk("Hartman3", [](){return new Hartman3();}, Vector::Constant(3,0.5), Hartman3::standardLower(), Hartman3::standardUpper(), Hartman3::globalMinimizer(), Hartman3::globalMinimum);
  mk("Hartman6", [](){return new Hartman6();}, Vector::Constant(6,0.5), Hartman6::standardLower(), Hartman6::standardUpper(), Hartman6::globalMinimizer(), Hartman6::globalMinimum);
  mk("Eggholder", [](){return new Eggholder();}, Vector{{0,0}}, Eggholder::standardLower(), Eggholder::standardUpper(), Eggholder::globalMinimizer(), Eggholder::globalMinimum);
  mk("Michalewicz n=5", [](){return new Michalewicz(5);}, Vector::Constant(5,1.5), Michalewicz::standardLower(5), Michalewicz::standardUpper(5), Vector::Constant(5,2.20319), -4.687658); // best known
  mk("Easom", [](){return new Easom();}, Vector{{0,0}}, Easom::standardLower(), Easom::standardUpper(), Easom::globalMinimizer(), Easom::globalMinimum);
  mk("ThreeHumpCamel", [](){return new ThreeHumpCamel();}, Vector{{2,1}}, ThreeHumpCamel::standardLower(), ThreeHumpCamel::standardUpper(), ThreeHumpCamel::globalMinimizer(), ThreeHumpCamel::globalMinimum);

  std::vector<BenchResult> results;
  std::cout << "\nEsecuzione " << cases.size() << " casi (Jamil13, Dixon78, MGH81 + legacy), 10 ripetizioni cadauno...\n\n";
  for(auto &c: cases){
    int ok_total=0, fail_total=0;
    int iter_min=INT_MAX, iter_max=0;
    double iter_sum=0.0;
    int conv_count=0;
    Scalar f_min_total = 0, f_max_total = 0;
    Scalar g_min_total = 0, g_max_total = 0;
    for(int rep=0; rep<10; ++rep){
      Vector x0_pert = c.x0;
      // Compute perturbation scale: proportional to box size or max(1,||x0||)
      double scale = 1.0;
      if(c.l.size() > 0 && c.u.size() > 0){
        // Use box size
        for(size_t j=0; j<c.x0.size(); ++j){
          scale = std::max(scale, std::abs(c.u(j) - c.l(j)));
        }
      } else {
        // Use norm of x0
        double norm_x0 = 0.0;
        for(size_t j=0; j<c.x0.size(); ++j) norm_x0 += c.x0(j)*c.x0(j);
        norm_x0 = std::sqrt(norm_x0);
        scale = std::max(1.0, norm_x0);
      }
      for(size_t j=0; j<c.x0.size(); ++j){
        double perturbation = (double)(rep+1) * scale * dist(rng);
        x0_pert(j) = c.x0(j) + perturbation;
      }
      Function* f=c.factory();
      Lbfgsb<> solver; solver.setAccuracy(0.9);
      if (g_verbose) {
        std::cout << "\n=== " << c.name << " (rep " << rep+1 << ") ===\n\n";
        // Print iter 0 initial values
        Scalar f0 = (*f)(x0_pert);
        Vector g0 = f->gradient(x0_pert);
        std::cout << "  iter=" << std::setw(4) << 0
                  << " feval=" << std::setw(5) << 1
                  << " geval=" << std::setw(5) << 1
                  << " f=" << std::fixed << std::setprecision(6) << std::setw(12) << f0
                  << " |g|=" << std::scientific << std::setprecision(2) << std::setw(10) << g0.lpNorm<Eigen::Infinity>()
                  << " df=---"
                  << " dx=---"
                  << "\n";
        solver.setCallback(verboseCallback);
      }
      State st = c.l.size()==0 ? solver.minimize(*f,x0_pert) : solver.minimize(*f,x0_pert,c.l,c.u);
      Scalar f_val = st.f();
      Scalar g_val = st.gNorm();
      if(st.success()) { ok_total++; conv_count++; iter_sum += (double)st.iter(); }
      else { fail_total++; }
      if(g_verbose) {
        if(st.success()) {
          std::cout << "  CONVERGED\n\n";
        } else {
          std::string reason;
          if(st.aborted()) reason = "ABORTED";
          else if(st.stalled()) reason = "STALLED";
          else if(st.stopped()) reason = "STOPPED";
          else reason = "FAILED";
          std::cout << "  " << reason
                    << " iter=" << std::setw(4) << st.iter()
                    << " feval=" << std::setw(5) << st.fEvals()
                    << " geval=" << std::setw(5) << st.gEvals()
                    << " f=" << std::fixed << std::setprecision(6) << std::setw(12) << st.f()
                    << " |g|=" << std::scientific << std::setprecision(2) << std::setw(10) << st.gNorm()
                    << " df=" << std::scientific << std::setprecision(2) << std::setw(10) << st.dfNorm()
                    << " dx=" << std::scientific << std::setprecision(2) << std::setw(10) << st.dxNorm()
                    << "\n\n";
        }
      }
      if(st.success()) {
        if(rep == 0 || f_val < f_min_total) f_min_total = f_val;
        if(rep == 0 || f_val > f_max_total) f_max_total = f_val;
        if(rep == 0 || g_val < g_min_total) g_min_total = g_val;
        if(rep == 0 || g_val > g_max_total) g_max_total = g_val;
      }
      if(st.iter() < iter_min) iter_min = st.iter();
      if(st.iter() > iter_max) iter_max = st.iter();
      delete f;
    }
    BenchResult r;
    r.name=c.name; r.n=(int)c.x0.size(); r.bounds=c.l.size()==0?"free":"box";
    r.iter_min = iter_min;
    r.iter_max = iter_max;
    r.iter_avg = conv_count > 0 ? iter_sum / conv_count : 0.0;
    r.f_min = f_min_total;
    r.f_max = f_max_total;
    r.g_min = g_min_total;
    r.g_max = g_max_total;
    r.ok_count = ok_total;
    r.fail_count = fail_total;
    r.success = (fail_total == 0);
    r.status = (r.ok_count == 10) ? "OK" : (r.ok_count > 0 ? std::to_string(r.ok_count)+"/10 OK" : "FAIL");
    results.push_back(r);
  }

  // ── Tabella unicode ───────────────────────────────────────────────────────
  const int Wnum=4, Wname=30, Wn=3, Witer=10, Wf=24, Wgn=24, Wstat=12;
  auto rep = [](const char* u,int n){ std::string s; for(int i=0;i<n;++i) s+=u; return s; };
  auto pad = [](std::string s, std::size_t w, bool left = true) {
    auto utf8_length = [](std::string_view str) {
      std::size_t n = 0;
      for (unsigned char c : str)
        if ((c & 0xC0) != 0x80) ++n;
      return n;
    };
    auto n = utf8_length(s);
    if (n > w) {
      std::size_t pos = 0;
      std::size_t count = 0;
      while (pos < s.size() && count < w) {
        ++pos;
        while (pos < s.size() && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) ++pos;
        ++count;
      }

      s.resize(pos);
      n = w;
    }
    auto spaces = w - n;
    return left
      ? s + std::string(spaces, ' ')
      : std::string(spaces, ' ') + s;
  };
  const std::string GREEN="\033[32m", YELLOW="\033[33m", RED="\033[31m", RESET="\033[0m";

  std::cout << "\n";
  std::cout << "╔" << rep("═",Wnum) << "╦" << rep("═",Wname) << "╦" << rep("═",Wn) << "╦" << rep("═",Witer) << "╦" << rep("═",Witer) << "╦" << rep("═",Witer) << "╦" << rep("═",Wf) << "╦" << rep("═",Wgn) << "╦" << rep("═",Wstat) << "╗\n";
  std::cout << "║" << pad(" #",Wnum,false) << "║" << pad(" Test",Wname) << "║" << pad(" n",Wn,false) << "║" << pad(" Iter min",Witer,false) << "║" << pad(" Iter max",Witer,false) << "║" << pad(" Iter avg",Witer,false) << "║" << pad(" f min / max",Wf,false) << "║" << pad(" g min / max",Wgn,false) << "║" << pad(" Conv",Wstat) << "║\n";
  std::cout << "╠" << rep("═",Wnum) << "╬" << rep("═",Wname) << "╬" << rep("═",Wn) << "╬" << rep("═",Witer) << "╬" << rep("═",Witer) << "╬" << rep("═",Witer) << "╬" << rep("═",Wf) << "╬" << rep("═",Wgn) << "╬" << rep("═",Wstat) << "╣\n";
  for(size_t idx=0; idx<results.size(); ++idx){
    auto &r = results[idx];
    std::ostringstream fmin_oss, fmax_oss;
    fmin_oss << std::scientific << std::setprecision(4) << r.f_min;
    fmax_oss << std::scientific << std::setprecision(4) << r.f_max;
    std::string fmin_str = r.ok_count > 0 ? fmin_oss.str() : "----";
    std::string fmax_str = r.ok_count > 0 ? fmax_oss.str() : "----";
    std::ostringstream gmin_oss, gmax_oss;
    gmin_oss << std::scientific << std::setprecision(2) << r.g_min;
    gmax_oss << std::scientific << std::setprecision(2) << r.g_max;
    std::string gmin_str = r.ok_count > 0 ? gmin_oss.str() : "----";
    std::string gmax_str = r.ok_count > 0 ? gmax_oss.str() : "----";
    std::string iter_min_str = r.ok_count > 0 ? std::to_string(r.iter_min) : "----";
    std::string iter_max_str = r.ok_count > 0 ? std::to_string(r.iter_max) : "----";
    std::ostringstream iter_avg_oss; iter_avg_oss << std::fixed << std::setprecision(1) << r.iter_avg;
    std::string iter_avg_str = r.ok_count > 0 ? iter_avg_oss.str() : "----";
    std::string conv_plain = (r.ok_count == 10) ? " ✓ OK" : (r.ok_count > 0 ? std::to_string(r.ok_count)+"/10 OK" : " ✗ FAIL");
    std::string conv_col = (r.ok_count == 10 ? GREEN : (r.ok_count > 0 ? YELLOW : RED)) + pad(conv_plain,Wstat) + RESET;
    // pad senza codici ANSI, poi colora
    std::cout << "║" << pad(std::to_string(idx+1),Wnum,false)
              << "║" << pad(" "+r.name,Wname)
              << "║" << pad(std::to_string(r.n),Wn,false)
              << "║" << pad(iter_min_str,Witer,false)
              << "║" << pad(iter_max_str,Witer,false)
              << "║" << pad(iter_avg_str,Witer,false)
              << "║" << pad(fmin_str + " / " + fmax_str,Wf,false)
              << "║" << pad(gmin_str + " / " + gmax_str,Wgn,false)
              << "║" << conv_col << "║\n";
  }
  std::cout << "╚" << rep("═",Wnum) << "╩" << rep("═",Wname) << "╩" << rep("═",Wn) << "╩" << rep("═",Witer) << "╩" << rep("═",Witer) << "╩" << rep("═",Witer) << "╩" << rep("═",Wf) << "╩" << rep("═",Wgn) << "╩" << rep("═",Wstat) << "╝\n";

  // Riepilogo
  int ok=0; for(auto &r:results) if(r.success) ++ok;
  std::cout << "\n» Summary: " << ok << "/" << results.size() << " converged ✓  |  "
            << "fails: " << (results.size()-ok) << "\n\n";
  return 0;
}
