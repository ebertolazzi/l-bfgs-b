// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License

#include "gtest/gtest.h"

#include "test.h"
#include "lbfgsb.hh"

using namespace optimize;
using LewisOvertonWeak = LewisOverton<Wolfe::weak>;
using LewisOvertonStrong = LewisOverton<Wolfe::strong>;

constexpr Scalar MAX_ERROR = 1e-5;
constexpr bool SHOW_RESULTS = false;

class SensitivityQuadratic : public Function
{
public:
  explicit SensitivityQuadratic(const Vector& target) : target(target) {}

  Scalar computeValue(const Vector& x) override
  { return 0.5*(x - target).squaredNorm(); }

  Vector computeGradient(const Vector& x) override
  { return x - target; }

private:
  Vector target;
};

class OneDimQuadratic : public Function
{
public:
  explicit OneDimQuadratic(const Scalar target) : target(target) {}

  Scalar computeValue(const Vector& x) override
  { return 0.5*(x(0) - target)*(x(0) - target); }

  Vector computeGradient(const Vector& x) override
  { return Vector {{ x(0) - target }}; }

  Matrix computeHessian(const Vector& /*x*/) override
  { return Matrix::Constant(1, 1, 1.0); }

private:
  Scalar target;
};

class ConcaveOneDim : public Function
{
public:
  Scalar computeValue(const Vector& x) override
  { return -x(0)*x(0); }

  Vector computeGradient(const Vector& x) override
  { return Vector {{ -2.0*x(0) }}; }

  Matrix computeHessian(const Vector& /*x*/) override
  { return Matrix::Constant(1, 1, -2.0); }
};

void minimizeSensitivityQuadratic(Lbfgsb<>& solver,
                                  const Vector& target,
                                  const Vector& l,
                                  const Vector& u
                                 )
{
  SensitivityQuadratic function(target);
  solver.minimize(function, Vector::Zero(target.size()), l, u);
}

TEST(SolverSensitivity, GradientPMinimizer)
{
  Lbfgsb<> solver;
  Matrix g_xx(2, 2);
  g_xx << 2.0, 0.0,
          0.0, 4.0;
  Matrix g_xp(2, 2);
  g_xp << 2.0, -4.0,
          8.0,  4.0;

  Matrix expected(2, 2);
  expected << -1.0,  2.0,
              -2.0, -1.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isApprox(expected));

  Simple function;
  Lbfgsb<> solver_with_active;
  solver_with_active.minimize(function,
                              Vector {{ 9.0, -8.0 }},
                              Vector {{-0.5, -10.0}},
                              Vector {{10.0,  10.0}}
                             );
  Matrix expected_with_active(2, 2);
  expected_with_active <<  0.0,  0.0,
                          -2.0, -1.0;
  EXPECT_TRUE(solver_with_active.gradientPMinimizer(g_xx, g_xp).isApprox(expected_with_active));
}

TEST(SolverSensitivity, GradientPMinimizerUpperBoundActive)
{
  Lbfgsb<> solver;
  minimizeSensitivityQuadratic(solver,
                               Vector {{ 5.0, 0.0 }},
                               Vector {{-10.0, -10.0}},
                               Vector {{ 1.0,  10.0}}
                              );
  Matrix g_xx(2, 2);
  g_xx << 2.0, 0.0,
          0.0, 4.0;
  Matrix g_xp(2, 2);
  g_xp << 2.0, -4.0,
          8.0,  4.0;
  Matrix expected(2, 2);
  expected <<  0.0,  0.0,
              -2.0, -1.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isApprox(expected));
}

TEST(SolverSensitivity, GradientPMinimizerAllBoundsActive)
{
  Lbfgsb<> solver;
  minimizeSensitivityQuadratic(solver,
                               Vector {{ 3.0, -4.0 }},
                               Vector {{-1.0, -2.0}},
                               Vector {{ 1.0,  2.0}}
                              );
  Matrix g_xx = Matrix::Identity(2, 2);
  Matrix g_xp(2, 3);
  g_xp << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isZero());
}

TEST(SolverSensitivity, GradientPMinimizerMultipleNonContiguousBounds)
{
  Lbfgsb<> solver;
  minimizeSensitivityQuadratic(solver,
                               Vector {{ 3.0, 0.0, -5.0, 0.0 }},
                               Vector {{-1.0, -10.0, -2.0, -10.0}},
                               Vector {{ 1.0,  10.0,  2.0,  10.0}}
                              );
  Matrix g_xx = Matrix::Zero(4, 4);
  g_xx.diagonal() << 2.0, 3.0, 4.0, 5.0;
  Matrix g_xp(4, 2);
  g_xp << 1.0,  2.0,
          6.0, -3.0,
          4.0,  8.0,
         10.0,  5.0;
  Matrix expected(4, 2);
  expected <<  0.0,  0.0,
              -2.0,  1.0,
               0.0,  0.0,
              -2.0, -1.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isApprox(expected));
}

TEST(SolverSensitivity, GradientPMinimizerMixedLowerAndUpperBounds)
{
  Lbfgsb<> solver;
  minimizeSensitivityQuadratic(solver,
                               Vector {{-4.0, 0.0, 5.0}},
                               Vector {{-1.0, -10.0, -10.0}},
                               Vector {{10.0,  10.0,   2.0}}
                              );
  Matrix g_xx = Matrix::Identity(3, 3);
  Matrix g_xp(3, 2);
  g_xp << 1.0, 2.0,
          3.0, 4.0,
          5.0, 6.0;
  Matrix expected(3, 2);
  expected <<  0.0,  0.0,
              -3.0, -4.0,
               0.0,  0.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isApprox(expected));
}

TEST(SolverSensitivity, GradientPMinimizerDenseHessian)
{
  Lbfgsb<> solver;
  Matrix g_xx(2, 2);
  g_xx << 4.0, 1.0,
          1.0, 3.0;
  Matrix g_xp(2, 2);
  g_xp << 5.0, 1.0,
          2.0, 4.0;
  Matrix expected(2, 2);
  expected << -13.0/11.0,   1.0/11.0,
               -3.0/11.0, -15.0/11.0;
  EXPECT_TRUE(solver.gradientPMinimizer(g_xx, g_xp).isApprox(expected));
}

TEST(SolverSensitivity, GradientPMinimizerRejectsInvalidDimensions)
{
  Lbfgsb<> solver;
  EXPECT_THROW(solver.gradientPMinimizer(Matrix::Identity(2, 2), Matrix::Zero(3, 1)), std::invalid_argument);
}

TEST(SolverSensitivity, GradientPMinimizerRejectsSingularFreeHessian)
{
  Lbfgsb<> solver;
  EXPECT_THROW(solver.gradientPMinimizer(Matrix::Zero(2, 2), Matrix::Ones(2, 1)), std::runtime_error);
}

TEST(SolverSensitivity, GradientPMinimum)
{
  Lbfgsb<> solver;
  const Vector g_p {{ 1.0, -2.0, 3.0 }};
  EXPECT_TRUE(solver.gradientPMinimum(g_p).isApprox(g_p));
}

TEST(SolverSensitivity, GradientPMinimumWithBoundedSolution)
{
  Lbfgsb<> solver;
  minimizeSensitivityQuadratic(solver,
                               Vector {{ 3.0, -4.0 }},
                               Vector {{-1.0, -2.0}},
                               Vector {{ 1.0,  2.0}}
                              );
  const Vector g_p {{-7.0, 0.0, 2.5}};
  EXPECT_TRUE(solver.gradientPMinimum(g_p).isApprox(g_p));
}

TEST(Minimize1D, NewtonStepFindsInteriorMinimum)
{
  OneDimQuadratic function(3.0);
  Minimize1D solver;
  const State& state = solver.minimize(function, -2.0, -10.0, 10.0);
  EXPECT_NEAR(state.x()(0), 3.0, MAX_ERROR);
  EXPECT_NEAR(state.f(), 0.0, MAX_ERROR);
  EXPECT_GT(state.hEvals(), 0);
}

TEST(Minimize1D, FindsBoundedMinimum)
{
  OneDimQuadratic function(-3.0);
  Minimize1D solver;
  const State& state = solver.minimize(function, 4.0, -1.0, 10.0);
  EXPECT_NEAR(state.x()(0), -1.0, MAX_ERROR);
  EXPECT_NEAR(state.f(), 2.0, MAX_ERROR);
}

TEST(Minimize1D, FallsBackFromNonPositiveCurvature)
{
  ConcaveOneDim function;
  Minimize1D solver;
  const State& state = solver.minimize(function, 0.5, -2.0, 2.0);
  EXPECT_NEAR(std::abs(state.x()(0)), 2.0, MAX_ERROR);
  EXPECT_NEAR(state.f(), -4.0, MAX_ERROR);
}

// Rosenbrock's banana-valley function in dimensions N=2,...,50.
// Reference: H. H. Rosenbrock, "An Automatic Method for Finding the Greatest or
// Least Value of a Function", The Computer Journal, 3(3), 175-184, 1960.
// This scalable, narrow curved-valley problem is also part of the numerical testing
// tradition documented by J. J. Moré, B. S. Garbow, K. E. Hillstrom, "Testing
// Unconstrained Optimization Software", ACM TOMS, 7(1), 17-41, 1981.
class RosenbrockDimensionsWeak : public ::testing::TestWithParam<Index> {};
class RosenbrockDimensionsStrong : public ::testing::TestWithParam<Index> {};

TEST_P(RosenbrockDimensionsWeak, ConvergesFromNarrowValley)
{
  const Index n = GetParam();
  Rosenbrock function;
  Lbfgsb<LewisOvertonWeak> solver;
  solver.setAccuracy(0.8);
  const State& state = solver.minimize(function,
                                       Vector::Constant(n, 0.8),
                                       Vector::Constant(n, -2.0),
                                       Vector::Constant(n, 2.0)
                                      );
  EXPECT_NEAR(state.f(), 0.0, MAX_ERROR);
}

TEST_P(RosenbrockDimensionsStrong, ConvergesFromNarrowValley)
{
  const Index n = GetParam();
  Rosenbrock function;
  Lbfgsb<LewisOvertonStrong> solver;
  solver.setAccuracy(0.8);
  const State& state = solver.minimize(function,
                                       Vector::Constant(n, 0.8),
                                       Vector::Constant(n, -2.0),
                                       Vector::Constant(n, 2.0)
                                      );
  EXPECT_NEAR(state.f(), 0.0, MAX_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
  Dimensions2To50,
  RosenbrockDimensionsWeak,
  ::testing::Values<Index>(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50)
);

INSTANTIATE_TEST_SUITE_P(
  Dimensions2To50,
  RosenbrockDimensionsStrong,
  ::testing::Values<Index>(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50)
);

#define LBFGSB_TEST_CASE(line_search, function, description, x, l, u, true_min)         \
  TEST(Lbfgsb##_##line_search, function##_##description) {                              \
    function f;                                                                         \
    Lbfgsb<line_search> solver;                                                         \
    Scalar solver_min = solver.minimize(f, Vector x, Vector l, Vector u).f();           \
    EXPECT_NEAR(solver_min, true_min, MAX_ERROR);                                       \
    EXPECT_TRUE(solver.state().success());                                              \
    if (SHOW_RESULTS) { std::cout << std::endl << solver << std::endl; }                \
  }

#define LBFGSB_TEST(function, description, x, l, u, true_min)                           \
  LBFGSB_TEST_CASE(LewisOvertonWeak, function, description, x, l, u, true_min)          \
  LBFGSB_TEST_CASE(LewisOvertonStrong, function, description, x, l, u, true_min)

// ============================================================================
// L-BFGS-B: Forrester tests
// ============================================================================
// Forrester: 0 variables bounded at minimum
LBFGSB_TEST(Forrester,      // Objective function
            0Active,        // Test Description
            ({{0.5241}}),   // Initial point
            ({{0}}),        // Lower bound
            ({{1}}),        // Upper bound
            -6.020740       // Expected minimum
           )

// Forrester: 1 variable bounded at minimum
LBFGSB_TEST(Forrester,      // Objective function
            1Active,        // Test Description
            ({{0.5241}}),   // Initial point
            ({{0}}),        // Lower bound
            ({{0.7}}),      // Upper bound
            -4.605754       // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Simple tests
// ============================================================================
// Simple: 0 variables bounded at minimum
LBFGSB_TEST(Simple,             // Objective function
            0Active,            // Test Description
            ({{   9,   -8}}),   // Initial point
            ({{ -10,  -10}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            -1.250000           // Expected minimum
           )

// Simple: 1 variable bounded at minimum: x0 = -0.5
LBFGSB_TEST(Simple,             // Objective function
            1Active,            // Test Description
            ({{   9,   -8}}),   // Initial point
            ({{-0.5,  -10}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            -1.000000           // Expected minimum
           )

// Simple: 2 variables bounded at minimum: x0 = -0.5, x1 = 2
LBFGSB_TEST(Simple,             // Objective function
            2Active,            // Test Description
            ({{   9,    6}}),   // Initial point
            ({{-0.5,    2}}),   // Lower bound
            ({{  10,   10}}),   // Upper bound
            0.000000            // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Non-Smooth Tests
// ============================================================================
// Non-Smooth 2D: 0 variables bounded at minimum
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 0Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,  -10}}),  // Lower bound
                 ({{  10,   10}}),  // Upper bound
                 0.000000           // Expected minimum
                )

// Non-Smooth 2D: 1 variable bounded at minimum: x0 = -5
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 1Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,  -10}}),  // Lower bound
                 ({{  -5,   10}}),  // Upper bound
                 2.924018           // Expected minimum
                )

// Non-Smooth 2D: 2 variables bounded at minimum: x0 = -5, x1 = 5
LBFGSB_TEST_CASE(LewisOvertonWeak,  // Line search (only Weak for non-smooth functions)
                 NonSmooth2D,       // Objective function
                 2Active,           // Test Description
                 ({{  -9,    8}}),  // Initial point
                 ({{ -10,    5}}),  // Lower bound
                 ({{  -5,   10}}),  // Upper bound
                 14.397915          // Expected minimum
                )

// ============================================================================
// L-BFGS-B: Rosenbrock tests
// ============================================================================
// Rosenbrock: 0 variables bounded at minimum
LBFGSB_TEST(Rosenbrock,           // Objective function
            0Active,              // Test Description
            ({{  8,  -5,   3}}),  // Initial point
            ({{-10, -10, -10}}),  // Lower bound
            ({{ 10,  10,  10}}),  // Upper bound
            0.000000              // Expected minimum
           )

// Rosenbrock: 1 variable bounded at minimum: x0 = 0.5
LBFGSB_TEST(Rosenbrock,               // Objective function
            1Active,                  // Test Description
            ({{   8,   -5,    3}}),   // Initial point
            ({{ -10,  -10,  -10}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            0.806931                  // Expected minimum
           )

// Rosenbrock: 2 variables bounded at minimum: x0 = 0.5, x1 = 0.5
LBFGSB_TEST(Rosenbrock,               // Objective function
            2Active,                  // Test Description
            ({{   0,    5,    5}}),   // Initial point
            ({{-0.5,  0.5,  -10}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            6.750000                  // Expected minimum
           )

// Rosenbrock: 3 variables bounded at minimum: x0 = 0.5, x1 = 0.5, x2 = 0.35
LBFGSB_TEST(Rosenbrock,               // Objective function
            3Active,                  // Test Description
            ({{   0,    5,    5}}),   // Initial point
            ({{-0.5,  0.5, 0.35}}),   // Lower bound
            ({{ 0.5,   10,   10}}),   // Upper bound
            7.750000                  // Expected minimum
           )

// Rosenbrock: out of bounds initial point
LBFGSB_TEST(Rosenbrock,               // Objective function
            OutOfBoundsInitialPoint,  // Test Description
            ({{  11,  -20,   30}}),   // Initial point
            ({{ -10,  -10,  -10}}),   // Lower bound
            ({{  10,   10,   10}}),   // Upper bound
            0.000000                  // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Six Hump Camel tests
// ============================================================================
// Six Hump Camel: 0 variables bounded at minimum
LBFGSB_TEST(SixHumpCamel,     // Objective function
            0Active,          // Test Description
            ({{  1,   0}}),   // Initial point
            ({{ -2,  -2}}),   // Lower bound
            ({{  2,   2}}),   // Upper bound
            -1.031628         // Expected minimum
           )

// Six Hump Camel: 1 variable bounded at minimum: x0 = 0.1
LBFGSB_TEST(SixHumpCamel,     // Objective function
            1Active,          // Test Description
            ({{  1,   0}}),   // Initial point
            ({{0.1,  -2}}),   // Lower bound
            ({{  2,   2}}),   // Upper bound
            -1.031230         // Expected minimum
           )

// Six Hump Camel: 2 variables bounded at minimum: x0 = 0.1, x1 = -0.6
LBFGSB_TEST(SixHumpCamel,     // Objective function
            2Active,          // Test Description
            ({{  1,    0}}),  // Initial point
            ({{0.1, -0.6}}),  // Lower bound
            ({{  2,    2}}),  // Upper bound
            -0.941810         // Expected minimum
           )

// ============================================================================
// L-BFGS-B: Spiral tests
// ============================================================================
// Spiral: 0 variables bounded at minimum
LBFGSB_TEST(Spiral,                   // Objective function
            0Active,                  // Test Description
            ({{   0,    0, 7.07}}),   // Initial point
            ({{-0.5, -0.5, 7.07}}),   // Lower bound
            ({{ 0.5,  0.5,  10}}),    // Upper bound
            0.313982                  // Expected minimum
           )

// Spiral: 1 variable bounded at minimum: x0 = 0.25
LBFGSB_TEST(Spiral,                     // Objective function
            1Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.25, 7.07}}),   // Lower bound
            ({{ 0.25,  0.25, 1e10}}),   // Upper bound
            6.832052                    // Expected minimum
           )

// Spiral: 2 variables bounded at minimum: x0 = 0.25, x1 = 0.2
LBFGSB_TEST(Spiral,                     // Objective function
            2Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.20, 7.07}}),   // Lower bound
            ({{ 0.25,  0.20, 1e10}}),   // Upper bound
            7.245711                    // Expected minimum
           )

// Spiral: 3 variables bounded at minimum: x0 = 0.25, x1 = 0.2, x2 = 8.0
LBFGSB_TEST(Spiral,                     // Objective function
            3Active,                    // Test Description
            ({{   0,    0,   7.07}}),   // Initial point
            ({{-0.25, -0.20, 7.07}}),   // Lower bound
            ({{ 0.25,  0.20, 8.00}}),   // Upper bound
            8.112968                    // Expected minimum
           )
