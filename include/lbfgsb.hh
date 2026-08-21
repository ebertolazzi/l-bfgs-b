#ifndef LBFGSB_HH
#define LBFGSB_HH

// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License


#include <iostream>    // cout, endl, ostream
#ifdef LBFGSB_USE_TIMEOUT
  #include <chrono>    // high_resolution_clock, duration
#endif
#include <limits>      // numeric_limits
#include <vector>      // vector

#include <Eigen/Core>  // Eigen

namespace optimize
{
  using Scalar = double;
  using Eigen::indexing::all;
  using Eigen::indexing::last;
  using Index = Eigen::Index;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using ScalarLimits = typename std::numeric_limits<Scalar>;
  using IndexLimits = typename std::numeric_limits<Index>;

#ifdef LBFGSB_USE_TIMEOUT
  using Clock = std::chrono::high_resolution_clock;
  using Time = Clock::time_point;
  using Duration = std::chrono::duration<Scalar>;
#endif

  class Function;
  struct State;

  // State of the solver
  struct SolverState
  {
    // Constructors and desctructors
    explicit SolverState(const Scalar df_norm = 0.0,  // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                         const Scalar dx_norm = 0.0,  // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                         const Scalar g_norm = 0.0,   // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                         const Index iter = 0,        // Iteration count
                         const Scalar duration = 0.0, // Duration (milliseconds)
                         const bool success = false,  // The solver successfully met the convergence criteria
                         const bool stopped = true,   // The solver is stopped but may be resumed
                         const bool aborted = false,  // The solver aborted the optimization and cannot be resumed
                         const bool stalled = false   // The solver is stalled (current x == previous x)
                        ) :
      df_norm(df_norm),
      dx_norm(dx_norm),
      g_norm(g_norm),
      iter(iter),
      duration(duration),
      success(success),
      stopped(stopped),
      aborted(aborted),
      stalled(stalled)
    {}

    Scalar df_norm;   // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
    Scalar dx_norm;   // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
    Scalar g_norm;    // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
    Index iter;       // Iteration count
    Scalar duration;  // Duration (milliseconds)
    bool success;     // The solver successfully met the convergence criteria
    bool stopped;     // The solver is stopped but may be resumed
    bool aborted;     // The solver aborted the optimization and cannot be resumed
    bool stalled;     // The solver is stalled (current x == previous x)
  };

  // Iterate state
  struct Iterate
  {
    // Constructors and desctructors
    explicit Iterate(const Vector& x = Vector(), // Parameter vector
                     const Scalar f = 0.0,       // Function value f(x)
                     const Vector& g = Vector(), // Gradient ▽f(x)
                     const Matrix& H = Matrix()  // Hessian ▽^2[f(x)]
                    ) :
      x(x),
      f(f),
      g(g),
      H(H)
    {}

    Vector x;   // Parameter vector
    Scalar f;   // Function value f(x)
    Vector g;   // Gradient ▽f(x)
    Matrix H;   // Hessian ▽^2[f(x)]
  };

  // Function data
  struct FunctionState
  {
    // Constructors and desctructors
    explicit FunctionState(const Index f_evals = 0,  // Number of function evaluations
                           const Index g_evals = 0,  // Number of gradient evaluations
                           const Index H_evals = 0   // Number of hessian evaluations
                          ) :
      f_evals(f_evals),
      g_evals(g_evals),
      H_evals(H_evals)
    {}

    Index f_evals;  // Number of function evaluations
    Index g_evals;  // Number of gradient evaluations
    Index H_evals;  // Number of hessian evaluations
  };

  // Stopping state for an optimization problem
  struct StopState
  {
    // Constructors and destructors
    explicit StopState(const SolverState& solver,   // Solver state
                       const Index f_evals = 0      // Number of function evaluations (0 = unlimited)
                      ) :
      solver(solver),
      f_evals(f_evals)
    {}

    explicit StopState(const Scalar& df_norm = 0.0, // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                       const Scalar& dx_norm = 0.0, // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                       const Scalar& g_norm = 0.0,  // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                       const Scalar duration = 0.0, // Duration (milliseconds) (0 = unlimited)
                       const Index f_evals = 0      // Number of function evaluations (0 = unlimited)
                      ) :
      solver(df_norm,  // df_norm
             dx_norm,  // dx_norm
             g_norm,   // g_norm
             0,        // iter
             duration  // duration
            ),
      f_evals(f_evals)
    {}

    // Solver state accessor methods
    const Scalar& dfNorm()   const { return solver.df_norm; }
    const Scalar& dxNorm()   const { return solver.dx_norm; }
    const Scalar& gNorm()    const { return solver.g_norm; }
    const Scalar& duration() const { return solver.duration; }

    Scalar& dfNorm()   { return solver.df_norm; }
    Scalar& dxNorm()   { return solver.dx_norm; }
    Scalar& gNorm()    { return solver.g_norm; }
    Scalar& duration() { return solver.duration; }

    // Function state accesor methods
    const Index& fEvals() const { return f_evals; }
    Index& fEvals() { return f_evals; }

    SolverState solver; // Solver state
    Index f_evals;      // Number of function evaluations (0 = unlimited)
  };

  // Full state of an optimization problem
  struct State
  {
    // Constructors and destructors
    explicit State(const Iterate& iterate,
                   const SolverState& solver = SolverState(),
                   const FunctionState& function = FunctionState()
                  ) :
      iterate(iterate),
      solver(solver),
      function(function)
    {}

    explicit State(const Vector& x = Vector(),  // Parameter vector
                   const Scalar f = 0.0,        // Function value f(x)
                   const Vector& g = Vector(),  // Gradient ▽f(x)
                   const Matrix& H = Matrix(),  // Hessian ▽^2[f(x)]
                   const Scalar df_norm = 0.0,  // Normalized delta in the function value, |fk+1 - fk|/max(|fk|, |fk+1|, 1)
                   const Scalar dx_norm = 0.0,  // Infinity norm of the delta in the state vector, L∞(xk+1 - xk)
                   const Scalar g_norm = 0.0,   // Infinity norm of the projected gradient, L∞(clip(xk+1 - ▽f(xk+1), l, u) - xk+1)
                   const Index iter = 0,        // Iteration count
                   const Scalar duration = 0.0, // Duration (milliseconds)
                   const bool success = false,  // The solver successfully met the convergence criteria
                   const bool stopped = true,   // The solver is stopped but may be resumed
                   const bool aborted = false,  // The solver aborted the optimization and cannot be resumed
                   const bool stalled = false,  // The solver is stalled (current x == previous x)
                   const Index f_evals = 0,     // Number of function evaluations
                   const Index g_evals = 0,     // Number of gradient evaluations
                   const Index H_evals = 0      // Number of hessian evaluations
                  ) :
      State(Iterate(x,  // x
                    f,  // f
                    g,  // g
                    H   // H
                   ),
            SolverState(df_norm,  // df_norm
                        dx_norm,  // dx_norm
                        g_norm,   // g_norm
                        duration, // duration
                        iter,     // iter
                        success,  // success
                        stopped,  // stopped
                        aborted,  // aborted
                        stalled   // stalled
                       ),
            FunctionState(f_evals,   // f_evals
                          g_evals,   // g_evals
                          H_evals    // H_evals
                         )
           )
    {}

    // Iterate state accessor functions
    const Vector& x() const { return iterate.x; }
    const Scalar& f() const { return iterate.f; }
    const Vector& g() const { return iterate.g; }
    const Matrix& H() const { return iterate.H; }

    Vector& x() { return iterate.x; }
    Scalar& f() { return iterate.f; }
    Vector& g() { return iterate.g; }
    Matrix& H() { return iterate.H; }

    // Solver state accessor methods
    const Scalar& dfNorm()   const { return solver.df_norm; }
    const Scalar& dxNorm()   const { return solver.dx_norm; }
    const Scalar& gNorm()    const { return solver.g_norm; }
    const Index& iter()      const { return solver.iter; }
    const Scalar& duration() const { return solver.duration; }
    const bool& success()    const { return solver.success; }
    const bool& stopped()    const { return solver.stopped; }
    const bool& aborted()    const { return solver.aborted; }
    const bool& stalled()    const { return solver.stalled; }

    Scalar& dfNorm()   { return solver.df_norm; }
    Scalar& dxNorm()   { return solver.dx_norm; }
    Scalar& gNorm()    { return solver.g_norm; }
    Index& iter()      { return solver.iter; }
    Scalar& duration() { return solver.duration; }
    bool& success()    { return solver.success; }
    bool& stopped()    { return solver.stopped; }
    bool& aborted()    { return solver.aborted; }
    bool& stalled()    { return solver.stalled; }

    // Function state accesor methods
    const Index& fEvals() const { return function.f_evals; }
    const Index& gEvals() const { return function.g_evals; }
    const Index& hEvals() const { return function.H_evals; }

    Index& fEvals() { return function.f_evals; }
    Index& gEvals() { return function.g_evals; }
    Index& hEvals() { return function.H_evals; }

    Iterate iterate;
    SolverState solver;
    FunctionState function;
  };

  // Prints an STL vector
  template <class T>
  inline std::ostream& operator<<(std::ostream& os, const std::vector<T> x)
  {
    for (size_t i = 0; i < x.size(); ++i) {
      if (i+1 == x.size()) {
        os << x[i];
      }
      else {
        os << x[i] << ", ";
      }
    }
    return os;
  }

  // Prints a solver state
  inline std::ostream& operator<<(std::ostream& os, const SolverState& state)
  {
    os << "Iterations = " << state.iter << std::endl;
    os << "Duration = " << state.duration << std::endl;
    os << "Success = " << (state.success ? "true" : "false") << std::endl;
    os << "df_norm = " << state.df_norm << std::endl;
    os << "dx_norm = " << state.dx_norm << std::endl;
    os << "g_norm = " << state.g_norm;
    return os;
  }

  // Prints an interate state
  inline std::ostream& operator<<(std::ostream& os, const Iterate& state)
  {
    os << "f = " << state.f << std::endl;
    os << "x = " << state.x.transpose() << std::endl;
    os << "g = " << state.g.transpose();
    for (Index i = 0; i < state.H.rows(); ++i) {
      if (i == 0) {
        os << std::endl << "H = " << state.H.row(i);
      }
      else {
        os << std::endl << "    " << state.H.row(i);
      }
    }
    return os;
  }

  // Prints a function state
  inline std::ostream& operator<<(std::ostream& os, const FunctionState& state)
  {
    os << "f_evals = " << state.f_evals << std::endl;
    os << "g_evals = " << state.g_evals << std::endl;
    os << "H_evals = " << state.H_evals;
    return os;
  }

  // Prints the full state of an optimization process
  inline std::ostream& operator<<(std::ostream& os, const State& state)
  {
    os << "Iterations = " << state.solver.iter << std::endl;
    os << "Duration = " << state.solver.duration << std::endl;
    os << "Success = " << (state.solver.success ? "true" : "false") << std::endl;
    os << state.iterate << std::endl;
    os << "df_norm = " << state.solver.df_norm << std::endl;
    os << "dx_norm = " << state.solver.dx_norm << std::endl;
    os << "g_norm = " << state.solver.g_norm << std::endl;
    os << state.function;
    return os;
  }

  // Clips a vector x to be within bounds [l, u]
  inline Vector clip(const Vector& x,
                     const Vector& l,
                     const Vector& u
                    )
  { return x.cwiseMin(u).cwiseMax(l); }

  // Clips a scalar value to be within bounds [l, u]
  inline Scalar clip(const Scalar x,
                     const Scalar l,
                     const Scalar u
                    )
  { return std::min(std::max(x, l), u); }

  // Shifts a matrix by the number of rows and columns specified, from a given starting row and column
  // Data past the end of the matrix which is shifted into the matrix is left as the old values (not reinitialized)
  template <class Derived>
  inline void shift(Eigen::MatrixBase<Derived>& x,
                    Index rows,
                    Index cols = 0,
                    Index row_start = 0,
                    Index col_start = 0
                   )
  {
    row_start = clip(row_start, 0, x.rows() - 1);
    col_start = clip(col_start, 0, x.cols() - 1);

    Index r_dir = rows > 0 ? -1 : 1;
    Index c_dir = cols > 0 ? -1 : 1;

    Index r_start = rows > 0 ? x.rows() - 1 : std::max(row_start + rows, static_cast<Index>(0));
    Index c_start = cols > 0 ? x.cols() - 1 : std::max(col_start + cols, static_cast<Index>(0));

    Index r = r_start;
    Index c = c_start;

    Index r_copy = r - rows;
    Index c_copy = c - cols;

    while (r_copy >= row_start && r_copy < x.rows()) {
      while (c_copy >= col_start && c_copy < x.cols()) {
        // Copy data to the shifted location
        x(r, c) = x(r_copy, c_copy);

        // Increment primary and runner column indexes
        c += c_dir;
        c_copy += c_dir;
      }
      // Increment primary and runner row indexes
      r += r_dir;
      r_copy += r_dir;

      // Reset column indexes for next row
      c = c_start;
      c_copy = c - cols;
    }
  }

  // Calculates the gradient projected onto the feasible region given by the bounds [l, u]
  // This represents how much x can change along the steepest descent direction when subject to the bounds.
  inline Vector projectedGradient(const Vector& x,
                                  const Vector& g,
                                  const Vector& l,
                                  const Vector& u
                                 )
  { return clip(x - g, l, u) - x; }

  // Returns the infinity norm of x
  template <class Derived>
  inline Scalar infinityNorm(const Eigen::MatrixBase<Derived>& x)
  { return x.template lpNorm<Eigen::Infinity>(); }

#ifdef LBFGSB_USE_TIMEOUT
  inline Scalar durationMsec(const Time& start, const Time& end)
  { return std::chrono::duration_cast<Duration>(end - start).count() * 1000.0; }
#endif

} // namespace optimize

// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License



namespace optimize
{
  // Function base class
  class Function
  {
  public:
    // Constructors and destructors
    Function() = default;
    virtual ~Function() = default;

    // Evaluates and returns an iterate state
    Iterate compute(const Vector& x)
    { return Iterate(x, this->operator()(x), gradient(x), hessian(x)); }

    // Returns the objective function value
    virtual Scalar operator()(const Vector& x)
    {
      function.f_evals++;
      return computeValue(x);
    }

    // Implementation of the objective function value
    virtual Scalar computeValue(const Vector& x) = 0;

    // Returns the gradient or subgradient at the point x
    Vector gradient(const Vector& x)
    {
      function.g_evals++;
      return computeGradient(x);
    }

    // Returns the gradient or subgradient at a given state
    Vector gradient(const Iterate& state)
    {
      function.g_evals++;
      return computeGradient(state);
    }

    // Implementation of the gradient or subgradient at the point x
    virtual Vector computeGradient(const Vector& x) = 0;

    // Implementation of the gradient or subgradient at a given state
    // Override this if you want to compute the gradient using previously calculated state information.
    // This is useful when elements of the gradient contain this state, and the state is expensive to compute.
    virtual Vector computeGradient(const Iterate& state)
    { return gradient(state.x); }

    // Returns the hessian at the point x
    Matrix hessian(const Vector& x)
    {
      function.H_evals++;
      return computeHessian(x);
    }

    // Returns the hessian at a given state
    Matrix hessian(const Iterate& state)
    {
      function.H_evals++;
      return computeHessian(state);
    }

    // Implementation of the hessian at the point x
    virtual Matrix computeHessian(const Vector& /*x*/)
    { return Matrix(); }

    // Implementation of the hessian at a given state
    // Override this if you want to compute the hessian using previously calculated state information.
    // This is useful when elements of the hessian contain this state, and the state is expensive to compute.
    virtual Matrix computeHessian(const Iterate& state)
    { return hessian(state.x); }

    // Returns the function state containing the number of evaluations computed
    const FunctionState& state() const
    { return function; }

    // Returns the current number of function evaluations computed
    const Index& fEvals() const
    { return function.f_evals; }

    // Returns the current number of gradient evaluations computed
    const Index& gEvals() const
    { return function.g_evals; }

    // Returns the current number of hessian evaluations computed
    const Index& hEvals() const
    { return function.H_evals; }

    // Resets info, clearing all evaluation counts
    void reset()
    {
      function.f_evals = 0;
      function.g_evals = 0;
      function.H_evals = 0;
    }

  private:
    FunctionState function; // Number of evaluations to f(x), g(x), and H(x)
  };

} // namespace optimize

// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License



namespace optimize
{
  // Wolfe condition enumeration
  enum Wolfe
  {
    weak,
    strong
  };

  // Line Search base class
  class LineSearch
  {
  public:
    // Constructors and desctructors
    explicit LineSearch(Index iter_max) :
      iter_max(iter_max)
    {}

    virtual ~LineSearch() = default;

    // Search method
    virtual Scalar operator()(Function& f,
                              const Scalar fx,
                              const Vector& x,
                              const Vector& g,
                              const Vector& d,
                              const Scalar t_max = ScalarLimits::max()
                             ) = 0;

  protected:
    const Index iter_max;
  };

  // A modification of the Lewis-Overton line search algorithm.
  // This class can be configured to enforce the weak or strong Wolfe condition on the directional gradient, and takes
  // a parameters for the maximum step size and iterations allowed.
  //
  // Reference: A. S. Lewis and M. L. Overton. "Nonsmooth optimization via quasi-Newton methods", Mathematical
  //            Programming, Vol 141, No 1, pp. 135-163, 2013
  template <Wolfe Condition>
  class LewisOverton : public LineSearch
  {
  private:
    static constexpr Wolfe condition = Condition;

  public:
    // Constructors and desctructors
    explicit LewisOverton(Index iter_max = 25) :
      LineSearch(iter_max)
    {}

    ~LewisOverton() = default;

    Scalar operator()(Function& f,
                      const Scalar fx,
                      const Vector& x,
                      const Vector& g,
                      const Vector& d,
                      const Scalar t_max = ScalarLimits::max()
                     ) override
    {
      // Initialize
      constexpr Scalar A = 1e-4;  // Armijo constant
      constexpr Scalar W = 0.9;   // Wolfe constant
      Index iter = 0;             // Iteration counter
      bool b_set = false;         // Interval upper limit has been set at least once this search
      bool t_ok = false;          // Step length passed both Armijo and Wolfe checks
      Scalar a = 0;               // Interval lower limit
      Scalar b = t_max;           // Interval upper limit
      Scalar h = 0.0;             // h(t) = f(x + t*d) - f(x)
      Scalar hp = 0.0;            // h'(t) = ▽f(x + t*d)^T*d
      Scalar s = g.dot(d);        // s = ▽f(x)^T*d = sup|t->0 {h(t)/t, t > 0}

      // If s is not sufficiently negative, d is not a useful descent direction
      // In this case we set t = 0 to skip the search and return t = 0
      Scalar t = s < -ScalarLimits::epsilon() && t_max > 0.0 ? clip(1.0, 0.0, t_max) : 0.0;
      Scalar t_prev = t;

      // Perform search
      while (   !t_ok
             && t > 0.0
             && (t < t_max || t_prev < t_max || iter == 0)
             && iter < this->iter_max
            ) {
        iter++;

        // Compute h(t) and h'(t)
        h = f(x + t*d) - fx;
        hp = f.gradient(x + t*d).dot(d);

        // Check Armijo condition: h(t) < A*s*t  ==>  f(x + t*d) - f(x) < A*▽f(x)^T*d*t
        // This requires the function value decreases proportional to the directional gradient times the step length
        if (h >= A*s*t) {
          b = t;
          b_set = true;
        }
        // Check strong Wolfe condition: |h'(t)| < |W*s|  ==>  |▽f(x + t*d)^T*d| < |W*▽f(x)^T*d|
        // This requires a sufficient decrease in the magnitude of the directional gradient, h'(t)
        else if (   condition == Wolfe::strong
                 && std::abs(hp) >= std::abs(W*s)
                 && !(hp < 0.0 && t == t_max)
                ) {
          if (hp < 0.0) {
            a = t;
          }
          else {
            b = t;
            b_set = true;
          }
        }
        // Check weak Wolfe condition: h'(t) > W*s  ==>  ▽f(x + t*d)^T*d > W*▽f(x)^T*d
        // This requires a sufficient increase in the directional gradient, h'(t)
        else if (   condition == Wolfe::weak
                 && hp <= W*s
                 && t != t_max
                ) {
          a = t;
        }
        // Both Armijo and strong Wolfe conditions were satisified
        else {
          t_ok = true;
        }

        if (!t_ok) {
          // Save previous t
          t_prev = t;

          // Update t
          // Interval upper limit has been set: set t at the midpoint of the new interval
          //
          // In the paper, the authors check b < ∞ to determine if the Armijo check has failed at least once and the
          // interval upper limit b has been set. If t_max < ∞ though, it is possible that the Armijo check does not
          // fail until exactly t == t_max, which would then set b = t == t_max and 'incorrectly' yield
          // b < t_max == false.
          // For this reason we directly check if the interval upper limit has been set with b_set.
          if (b_set) {
            t = (a + b)/2;
          }
          // Interval upper limit has not been set: keep increasing t until max is reached
          else if (t < t_max) {
            t = 2*a;
          }

          // Clip t to be within [0, t_max]
          t = clip(t, 0.0, t_max);
        }
      }

      // Return 0 if a suitable step length could not be found and allow the solver to decide how to handle this
      return t_ok ? t : 0.0;
    }
  };

} // namespace optimize

// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License


#include <cmath>      // pow
#include <iostream>   // cout, endl
#include <functional> // function
#include <stdexcept>  // invalid_argument, runtime_error

#include <Eigen/LU>   // FullPivLU


namespace optimize
{
  // Solver base class
  // Executes the high level optimization procedure, calling on the derived class to perform the optimization step
  // which computes a new state.
  class Solver
  {
  public:
    using Callback = std::function<void (Solver*)>;

  public:
    // Constructors and destructors
    explicit Solver(const Callback& callback = [](Solver*) {}) :
      n(0),
      l(Vector::Zero(n)),
      u(Vector::Zero(n)),
      curr_state(),
      prev_state(),
      stop_state(),
      callback(callback)
    { setAccuracy(default_accuracy); }

    virtual ~Solver() = default;

    // Friend operator <<
    friend std::ostream& operator<<(std::ostream& os, const Solver& solver);

    // Initializes the state for each new optimization
    void initialize(Function& f,
                    const Iterate& iterate,
                    const Vector& l = Vector(),
                    const Vector& u = Vector()
                   )
    {
#ifdef LBFGSB_USE_TIMEOUT
      start_time = Clock::now();  // Reset the start time
#endif
      n = iterate.x.size();       // Save the size of the parameter vector x

      // Form constraints as unbounded if not specified
      this->l = l.size() == 0 ? Vector::Constant(n, ScalarLimits::lowest()) : l;
      this->u = u.size() == 0 ? Vector::Constant(n, ScalarLimits::max()) : u;

      // Resize l and u if necessary, handling the case if the user specified l and u with sizes that don't match x
      Scalar l_size = this->l.size();
      Scalar u_size = this->u.size();
      this->l.conservativeResize(n);
      this->u.conservativeResize(n);

      // If l or u increased in size, populate them with min/max bounds
      for (Index i = l_size; i < n; ++i) {
        this->l(i) = ScalarLimits::lowest();
      }
      for (Index i = u_size; i < n; ++i) {
        this->u(i) = ScalarLimits::max();
      }

      // Flip constraints if l(i) > u(i)
      for (Index i = 0; i < n; ++i) {
        Scalar li = this->l(i);
        this->l(i) = std::min<Scalar>(this->l(i), this->u(i));
        this->u(i) = std::max<Scalar>(li,         this->u(i));
      }

      // Initialize the solver state with the initial iterate, enforcing that x is within bounds [l, u]
      curr_state = State(iterate);
      curr_state.x() = clip(curr_state.x(), this->l, this->u);
      prev_state = curr_state;

      f.reset();              // Reset the number of function calls made
      reset();                // Reset the algorithm's internal data
      updateState(f, false);  // Compute initial solver state with convergence data
    }

    void initialize(Function& f,
                    const Vector& x,
                    const Vector& l = Vector(),
                    const Vector& u = Vector()
                   )
    { initialize(f, f.compute(x), l, u); }

    // Minimizes the function f starting from x and subject to bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize(Function& f,
                          const Vector& x,
                          const Vector& l = Vector(),
                          const Vector& u = Vector()
                         )
    {
      initialize(f, x, l, u); // Initialize the solver for the new minimization problem
      return minimize(f);     // Minimize f from the initial state and return the result
    }

    // Minimizes the function f starting from the specified function state and subject to bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize(Function& f,
                          const Iterate& iterate,
                          const Vector& l = Vector(),
                          const Vector& u = Vector()
                         )
    {
      initialize(f, iterate, l, u); // Initialize the solver for the new minimization problem
      return minimize(f);           // Minimize f from the initial state and return the result
    }

    // Minimizes the function f from the current state with previously-specified bound constraints l and u.
    // Returns the full state when the convergence criteria is met or the solver is stopped.
    const State& minimize(Function& f)
    {
      curr_state.stopped() = false; // Reset the stopped flag to allow the solver to run
#ifdef LBFGSB_USE_TIMEOUT
      start_time = Clock::now();    // Reset the start time
#endif

      // Compute a new state until the solver is done or stopped by the user
      while (!done() && !stopped()) {
        performStep(f); // Perform the optimization step to compute a new iterate
        updateState(f); // Update the solver state with the current iterate
        callback(this); // Execute the user callback function
      }

      return curr_state;  // Return the solution containing the minimum
    }

    // Stops the solver at the current state after the current optimization step is complete.
    // The solver may be resumed at this state by subsequently calling run().
    void stop()
    { curr_state.stopped() = true; }

    // Sets the solver convergence criteria based on the desired level of accuracy.
    // Accuracy level ranges from 0 to 1, where 1 is maximum accuracy possible based on the machine epsilon of the
    // Scalar type.
    void setAccuracy(Scalar accuracy)
    {
      // Clip accuracy to be within [0, 1]
      accuracy = clip(accuracy, 0.0, 1.0);

      // Compute converge criteria based on the machine epsilon.
      // The highest achievable accuracy in the objective value or parameter vector is the machine epsilon.
      // The target gradient norm is recommended to be at least sqrt(epsilon) by the authors of the L-BFGS-B paper.
      stop_state.dfNorm() = std::pow(10, accuracy*log10(ScalarLimits::epsilon()));
      stop_state.dxNorm() = std::pow(10, accuracy*log10(ScalarLimits::epsilon()));
      stop_state.gNorm() = std::pow(10, accuracy*log10(std::sqrt(ScalarLimits::epsilon())));
    }

    // Sets the maximum number of function evaluations before the solver stops (0 = unlimited).
    void setMaxFunctionEval(const Index f_evals_max)
    { stop_state.fEvals() = f_evals_max; }

#ifdef LBFGSB_USE_TIMEOUT
    // Sets the maximum execution time in milliseconds (0 = unlimited).
    void setTimeout(const Scalar ms)
    { stop_state.duration() = ms; }
#endif

    // Sets the callback function
    void setCallback(const Callback& callback)
    { this->callback = callback; }

    // Computes the Jacobian dX*/dP of a locally optimal solution X*(P).
    //
    // At an unconstrained stationary point, differentiating g_X(X*(P), P) = 0 gives
    //
    //   g_XX dX*/dP = -g_XP,
    //
    // and therefore dX*/dP = -g_XX^-1 g_XP. The arguments must be evaluated at the
    // computed optimum: g_xx has shape (n, n) and g_xp has shape (n, np), where np
    // is the number of parameters. The returned matrix has shape (n, np).
    //
    // For parameter-independent box constraints, the active set is obtained from the
    // solution and bounds stored by the last call to minimize(). Its sensitivities are
    // set to zero and the system is solved only on the complementary free variables.
    // This is valid locally only while the active set does not change. Bounds depending
    // on P require their own derivative terms and are not handled by this method.
    Matrix gradientPMinimizer(const Matrix& g_xx,
                              const Matrix& g_xp
                             ) const
    {
      if (g_xx.rows() != g_xx.cols() || g_xp.rows() != g_xx.rows()) {
        throw std::invalid_argument("g_xx must be square and g_xp must have the same number of rows");
      }

      const Index n = g_xx.rows();
      std::vector<bool> is_active(static_cast<size_t>(n), false);
      for (const Index i : sensitivity_active_set) {
        if (i < 0 || i >= n) {
          throw std::invalid_argument("the stored active set is incompatible with g_xx");
        }
        is_active[static_cast<size_t>(i)] = true;
      }

      std::vector<Index> free_set;
      free_set.reserve(static_cast<size_t>(n));
      for (Index i = 0; i < n; ++i) {
        if (!is_active[static_cast<size_t>(i)]) {
          free_set.push_back(i);
        }
      }

      Matrix gradient = Matrix::Zero(n, g_xp.cols());
      if (free_set.empty()) {
        return gradient;
      }

      const Index n_free = static_cast<Index>(free_set.size());
      Matrix hessian_free(n_free, n_free);
      Matrix mixed_free(n_free, g_xp.cols());
      for (Index i = 0; i < n_free; ++i) {
        mixed_free.row(i) = g_xp.row(free_set[static_cast<size_t>(i)]);
        for (Index j = 0; j < n_free; ++j) {
          hessian_free(i, j) = g_xx(free_set[static_cast<size_t>(i)], free_set[static_cast<size_t>(j)]);
        }
      }

      Eigen::FullPivLU<Matrix> decomposition(hessian_free);
      if (!decomposition.isInvertible()) {
        throw std::runtime_error("the free-variable Hessian g_xx is singular");
      }

      const Matrix gradient_free = -decomposition.solve(mixed_free);
      for (Index i = 0; i < n_free; ++i) {
        gradient.row(free_set[static_cast<size_t>(i)]) = gradient_free.row(i);
      }
      return gradient;
    }

    // Computes the gradient d f_min/dP of the minimum value f_min(P) = g(X*(P), P).
    //
    // The envelope theorem gives d f_min/dP = g_P(X*(P), P): the derivative of X*(P)
    // does not enter because the first-order optimality conditions cancel it. Thus g_p
    // must be evaluated at the computed optimum and is returned unchanged. This result
    // assumes differentiability and parameter-independent box constraints; if a bound
    // depends on P, the corresponding KKT multiplier and bound derivative are needed.
    Vector gradientPMinimum(const Vector& g_p) const
    { return g_p; }

    // Returns the current state
    const State& state() const
    { return curr_state; }

    // Returns true if the solver met the convergence criteria
    bool success()
    { return curr_state.success(); }

    // Returns true if the solver is stopped
    bool stopped()
    { return curr_state.stopped(); }

    // Returns true if the solver has aborted the optimization due to a termination condition being met or, for an
    // algorithm-specific reason, further progress cannot be made. This may or may not indicate failure; in general it
    // means that the algorithm is unable to improve from the current state.
    bool aborted()
    { return curr_state.aborted(); }

    // Returns true if the solver cannot make further progress because a convergence or termination condition was met
    bool done()
    { return success() || aborted(); }

  protected:
    // Resets the algorithm's internal data and restarts the optimization at the current iterate
    virtual void reset() {}

    // Performs one optimization step, updating the current and previous iterate
    virtual void performStep(Function& f) = 0;

    // Updates the solver state with the current iterate
    void updateState(Function& f, bool post_step = true)
    {
      // If the current x is exactly the same as the previous, the algorithm has stalled
      const Scalar dx_max = infinityNorm(curr_state.x() - prev_state.x());
      curr_state.stalled() = dx_max == 0.0;

      // Update convergence data
      // Skip calculating df and dx norms if current x == previous x
      if (!curr_state.stalled()) {
        curr_state.dfNorm() =   std::abs(curr_state.f() - prev_state.f())
                              / std::max(std::max(std::abs(curr_state.f()),
                                                  std::abs(prev_state.f())
                                                 ),
                                         static_cast<Scalar>(1.0)
                                        );
        curr_state.dxNorm() =   dx_max
                              / std::max(std::max(infinityNorm(curr_state.x()),
                                                  infinityNorm(prev_state.x())
                                                 ),
                                         static_cast<Scalar>(1.0)
                                        );
      }
      curr_state.gNorm() = infinityNorm(projectedGradient(curr_state.x(), curr_state.g(), l, u));
      updateSensitivityActiveSet();

      // Update duration only when timeout support is enabled.
#ifdef LBFGSB_USE_TIMEOUT
      end_time = Clock::now();
      curr_state.duration() += durationMsec(start_time, end_time);
      start_time = end_time;
#endif

      // Copy function data
      curr_state.function = f.state();

      // Check stopping criteria
      const bool df_min_success = (curr_state.dfNorm() <= stop_state.dfNorm() && !curr_state.stalled());
      const bool dx_min_success = (curr_state.dxNorm() <= stop_state.dxNorm() && !curr_state.stalled());
      const bool g_min_success = (curr_state.gNorm() <= stop_state.gNorm());
#ifdef LBFGSB_USE_TIMEOUT
      const bool duration_exceeded = (curr_state.duration() >= stop_state.duration() && stop_state.duration() > 0.0);
#else
      constexpr bool duration_exceeded = false;
#endif
      const bool f_evals_exceeded = (curr_state.fEvals() >= stop_state.fEvals() && stop_state.fEvals() > 0);

      // Update solver statuses
      curr_state.success() = (   df_min_success
                              || dx_min_success
                              || g_min_success
                             );
      curr_state.aborted() = (   curr_state.aborted()
                              || (curr_state.stalled() && prev_state.stalled())
                              || duration_exceeded
                              || f_evals_exceeded
                             );
      curr_state.iter() = post_step ? curr_state.iter() + 1 : curr_state.iter();

      // Update previous state
      prev_state = curr_state;
    }

  protected:
    // Determines which variables are fixed at a bound in the current solution. This
    // active set is intentionally independent from L-BFGS-B's Cauchy-point active set.
    void updateSensitivityActiveSet()
    {
      sensitivity_active_set.clear();
      const Scalar tolerance = 100*ScalarLimits::epsilon();
      for (Index i = 0; i < n; ++i) {
        const bool has_lower_bound = l(i) != ScalarLimits::lowest();
        const bool has_upper_bound = u(i) != ScalarLimits::max();
        const bool at_lower_bound = has_lower_bound
                                  && std::abs(curr_state.x()(i) - l(i))
                                     <= tolerance*std::max<Scalar>(1.0, std::abs(l(i)));
        const bool at_upper_bound = has_upper_bound
                                  && std::abs(curr_state.x()(i) - u(i))
                                     <= tolerance*std::max<Scalar>(1.0, std::abs(u(i)));
        if (at_lower_bound || at_upper_bound) {
          sensitivity_active_set.push_back(i);
        }
      }
    }

    static constexpr Scalar default_accuracy = 0.7;

    Index n;              // Size of x
    Vector l;             // Lower bounds for x
    Vector u;             // Upper bounds for x
    State curr_state;     // Current state
    State prev_state;     // Previous state
    std::vector<Index> sensitivity_active_set; // Bound-active indices at the current solution
#ifdef LBFGSB_USE_TIMEOUT
    Time end_time;        // Current time point
    Time start_time;      // Previous time point
#endif
    StopState stop_state; // Stopping state
    Callback callback;    // Optional user-defined callback function
  };

  // Prints the solver state including termination messages, if any
  inline std::ostream& operator<<(std::ostream& os, const Solver& solver)
  {
    const State& curr_state = solver.curr_state;
    const StopState& stop_state = solver.stop_state;

    std::cout << curr_state;

    if (curr_state.success()) {
      std::cout << std::endl;
      if (curr_state.dfNorm() <= stop_state.dfNorm()) {
        std::cout << std::endl << "Change in f is below threshold";
      }
      if (curr_state.dxNorm() <= stop_state.dxNorm()) {
        std::cout << std::endl << "Change in x is below threshold";
      }
      if (curr_state.gNorm() <= stop_state.gNorm()) {
        std::cout << std::endl << "Gradient norm is below threshold";
      }
    }
    else {
#ifdef LBFGSB_USE_TIMEOUT
      const bool duration_exceeeded = (   curr_state.duration() >= stop_state.duration()
                                       && stop_state.duration() > 0.0
                                      );
#else
      constexpr bool duration_exceeeded = false;
#endif
      const bool f_evals_exceeded = (   curr_state.fEvals() >= stop_state.fEvals()
                                     && stop_state.fEvals() > 0
                                    );
      if (   duration_exceeeded
          || f_evals_exceeded
          || curr_state.aborted()
          || curr_state.stopped()
         ) {
        std::cout << std::endl;
        if (duration_exceeeded) {
          std::cout << std::endl << "Maxmimum execution time exceeded";
        }
        if (f_evals_exceeded) {
          std::cout << std::endl << "Maxmimum number of function evaluations reached";
        }
        if (curr_state.aborted()) {
          std::cout << std::endl << "Optimization aborted by algorithm";
        }
        if (curr_state.stopped()) {
          std::cout << std::endl << "Optimization stopped by user";
        }
      }
    }
    return os;
  }

} // namespace optimize

// Copyright (c) 2023 Dane Roemer droemer7@gmail.com
// Distributed under the terms of the MIT License


#include <cassert>          // assert
#include <iostream>         // cout, endl, ostream
#include <vector>           // vector

#include <Eigen/LU>         // lu()


namespace optimize
{
  // Breakpoint for the generalized Cauchy Point search
  struct Breakpoint
  {
    explicit Breakpoint(const Index i = -1, const double t = 0.0) :
      i(i),
      t(t)
    {}

    bool operator==(const Breakpoint& rhs) const
    { return this->t == rhs.t; }

    bool operator!=(const Breakpoint& rhs) const
    { return this->t != rhs.t; }

    bool operator<=(const Breakpoint& rhs) const
    { return this->t <= rhs.t; }

    bool operator<(const Breakpoint& rhs) const
    { return this->t < rhs.t; }

    bool operator>=(const Breakpoint& rhs) const
    { return this->t >= rhs.t; }

    bool operator>(const Breakpoint& rhs) const
    { return this->t > rhs.t; }

    Index i;  // Index of the variable in x
    double t; // Breakpoint for the variable i in x
  };

  inline std::ostream& operator<<(std::ostream& os, const std::vector<Breakpoint>& breakpoints)
  {
    os << "breakpoints = ";
    if (breakpoints.size() == 0) {
      os << std::endl;
    }
    else {
      for (size_t i = 0; i < breakpoints.size(); ++i) {
        os << std::endl << "t for x(" << breakpoints[i].i << ") = " << breakpoints[i].t;
      }
    }
    return os;
  }

  // L-BFGS-B: Limited-memory BFGS algorithm for bound constrained optimization
  //
  // Reference: R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A Limited Memory Algorithm for Bound Constrained
  //            Optimization", Tech. Report, NAM-08, EECS Department, Northwestern University, 1994.
  template <class LineSearch = LewisOverton<Wolfe::weak>>
  class Lbfgsb : public Solver
  {
  public:
    using IndexSet = std::vector<Index>;
    using BreakpointSet = std::vector<Breakpoint>;
    using Solver::Callback;

  public:
    // Constructors and destructors
    explicit Lbfgsb(const Callback& callback = [](Solver*) {}) :
      Solver(callback)
    {}

  private:
    // Resets the algorithm's internal data
    void reset() override
    {
      // Initialize BFGS information and matrices
      m = 0;
      th = 1.0;
      th_inv = 1.0;

      I = Matrix::Identity(n, n);
      S = Matrix::Zero(n, 1);
      Y = Matrix::Zero(n, 1);
      SS = Matrix::Zero(1, 1);
      SY = Matrix::Zero(1, 1);
      YY = Matrix::Zero(1, 1);

      D = Matrix::Zero(1, 1);
      R_inv = Matrix::Zero(1, 1);

      W = Matrix::Zero(n, 2);
      Wb = Matrix::Zero(n, 2);

      M = Matrix::Zero(2, 2);
      Mb = Matrix::Zero(2, 2);

      c = Vector::Zero(2);

      free_set.reserve(n);
      free_set.clear();
      active_set.reserve(n);
      active_set.clear();
    }

    // Performs one optimization step of the algorithm, updating the current and previous iterate
    void performStep(Function& f) override
    {
      // Create aliases for readability
      Iterate& prev = prev_state.iterate;
      Iterate& curr = curr_state.iterate;

      Vector xc = cauchyPoint(curr.x, curr.g, l, u);                // Compute the generalized Cauchy point xc
      Vector d = searchDir(xc, curr.x, curr.g, l, u);               // Compute the search direction d
      Scalar t_max = maxStep(curr.x, l, u, d);                      // Compute the max step possible along d
      Scalar t = line_search(f, curr.f, curr.x, curr.g, d, t_max);  // Compute the step to take along d

      // A suitable step was found: compute the next iterate and update the limited memory matrices
      if (t > 0.0) {
        prev = curr;                      // Save the previous iterate
        curr.x += t*d;                    // Compute the next iterate x = x + t*d
        curr.f = f(curr.x);               // Compute the function value f(x)
        curr.g = f.gradient(curr.x);      // Compute the gradient ▽f(x)
        updateMatrices(curr.x - prev.x,   // Update the limited memory matrices
                       curr.g - prev.g
                      );
      }
      // A suitable step was not found: discard all correction pairs and restart along the steepest descent direction
      else {
        reset();
      }
    }

    // Determines the first local minimizer of the univariate, piecewise quadratic q(t):
    //
    //   q(t) = m(P(xk - t*g, l u))
    //
    // where
    //   m(x) = f(xk) + ▽f(xk)^T(x - xk) + 1/2(x - xk)^T*Bk*(x - xk)
    //   P(x, l, u)(i) = { l(i)  if x(i) < l(i)
    //                   { x(i)  if x(i) is in [l(i), u(i)]
    //                   { u(i)  if x(i) > u(i)
    //   Bk = th*I - W*M*W^T
    Vector cauchyPoint(const Vector& x,
                       const Vector& g,
                       const Vector& l,
                       const Vector& u
                      )
    {
      // Initialize the search
      // Set xc = x to start - we will modify xc if/when appropriate during the search
      Vector xc = x;
      BreakpointSet breakpoints;
      Vector t = Vector::Zero(n);
      Vector d = Vector::Zero(n);
      Vector p = Vector::Zero(W.cols());
      Vector w = Vector::Zero(W.cols());
      c = Vector::Zero(W.cols());
      free_set.clear();
      active_set.clear();

      size_t q = 0;
      Index b = -1;
      Scalar fp = 0;
      Scalar fpp = 0;
      Scalar dt_min = 0;
      Scalar t_start = 0;
      Scalar t_end = 0;
      Scalar dt = 0;
      Scalar t_min = 0;
      Scalar zb = 0;

      // Calculate breakpoints t(i) and descent directions d(i)
      for (Index i = 0; i < n; ++i) {
        // Calculate breakpoints t(i)
        if (g(i) < 0) {
          t(i) = (x(i) - u(i))/g(i);
        }
        else if (g(i) > 0) {
          t(i) = (x(i) - l(i))/g(i);
        }
        else {
          t(i) = ScalarLimits::max();
        }
        // Calculate descent directions d(i) for variables with breakpoints t(i) > 0
        // Otherwise, t(i) == 0 so we leave d(i) = 0 and xc(i) = x(i) from initialization above
        if (t(i) > 0) {
          d(i) = -g(i);
          breakpoints.push_back(Breakpoint(i, t(i)));
        }
      }

      if (breakpoints.size() > 0) {
        // Sort indexes i by ascending breakpoint value t
        std::sort(breakpoints.begin(), breakpoints.end());

        // Initialize the storage vector p used to update f' and f''
        p = W.transpose()*d;

        // Calculate the initial dt_min
        fp = -d.dot(d);
        fpp = -th*fp - p.transpose()*M*p;
        dt_min = (fpp == 0) ? ScalarLimits::max() : -fp/fpp;

        // Define the first interval
        b = breakpoints[q].i;
        t_start = t_end;
        t_end = t(b);
        dt = t_end - t_start;

        // Search for the step to the first minimum along the steepest descent direction d(i) = -g(i)
        // Inspect each interval in t until dt_min is less than or equal to the interval
        while (dt_min > dt) {
          assert (d(b) != 0);

          // Since dt_min > dt, the current component's minimum is not in the interval and must be bounded
          // Accordingly, we also zero out the current component's search direction
          xc(b) = d(b) > 0 ? u(b) : l(b);
          zb = xc(b) - x(b);
          d(b) = 0;

          // Update vector c which will be used to initialize the subspace minimization problem
          c += dt*p;

          // Calculate f' and f'' for determining the location of the minimum
          w = W(b, Eigen::indexing::all).transpose();
          fp += dt*fpp + g(b)*g(b) + th*g(b)*zb - g(b)*w.transpose()*M*c;
          fpp += -th*g(b)*g(b) - 2*g(b)*w.transpose()*M*p - g(b)*g(b)*w.transpose()*M*w;
          dt_min = (fpp == 0) ? ScalarLimits::max() : -fp/fpp;

          // Update p for the next calculation of f' and f''
          p += g(b)*w;

          // Update to the next interval
          t_start = t_end;

          if (++q < breakpoints.size()) {
            b = breakpoints[q].i;
            t_end = t(b);
            dt = t_end - t_start;
          }
          // If there are no more intervals, the previous iteration determined that the final component of x(b=n) is
          // bounded. In this case dt_min will be past the final breakpoint t(b=n) and there will be no free variables.
          else {
            t_end = t_start;
            dt = 0;
            break;
          }
        }
      }
      // Calculate t_min
      dt_min = std::max<Scalar>(dt_min, 0);
      t_min = t_start + dt_min;

      // Construct the sets of free and active variable indexes i and calculate the remaining components xc(i) of the
      // Cauchy point from t_min.
      //
      // Note: There is an error in the paper here: t_min should be used instead of t_end. The final steps specify
      //       updating xc(i) for t(i) >= t_end and removing i from the free set if t(i) == t_end. This would mean
      //       removing indexes for xc(i) that are NOT at their bound because usually t_min < t_end, and
      //       xc(i) = x(i) + t_min*d(i). Since the minimum is found at t_min, we should compute all xc(i) for
      //       t(i) >= t_min and then remove i from F if t(i) == t_min, because that means xc(i) will be at its bound.
      for (Index i = 0; i < n; ++i) {
        // If q is past the end of the breakpoint array, there are no free variables.
        // Either: 1) There were no breakpoints t(i) > 0, so all xc(i) are already bounded (and xc == x from
        //            initialization)
        //         2) The Cauchy point lies beyond the largest breakpoint t(i) in x, in which case all xc(i) were set to
        //            their bound u(i) or l(i) during the search.
        if (q < breakpoints.size()) {
          // Compute xc(i) using the step t_min to the first minimum found earlier
          if (t(i) >= t_min) {
            xc(i) = x(i) + t_min*d(i);
          }

          // Add i to the free set if x(i) is unbounded (its breakpoint t(i) lies beyond the minimum t_min)
          // Otherwise, add i to the active set
          if (t(i) > t_min) {
            free_set.push_back(i);
          }
          else {
            active_set.push_back(i);
          }
        }
        else {
          active_set.push_back(i);
        }
      }
      c += dt_min*p;  // Update vector c for the subspace minimization problem

      return xc;
    }

    // Approximately solves for the search direction d of the subspace minimization problem m(d):
    //
    //   m(d) = d^T*rc + 1/2(d^T*B*d) + y
    //   subject to l(i) - xc(i) <= d(i) <= u(i) - xc(i)
    //
    // where
    //   rc is the reduced gradient of m at xc
    //   B is the reduced hessian of m
    Vector searchDir(const Vector& xc,
                     const Vector& x,
                     const Vector& g,
                     const Vector& l,
                     const Vector& u
                    )
    {
      Matrix A = I(all, active_set);  // Matrix of unit vectors spanning the active set at the Cauchy point xc (n x ta)
      Matrix Z = I(all, free_set);    // Matrix of unit vectors spanning the free set at the Cauchy point xc (n x tf)
      Vector v;                       // Vector v (2m x 1)
      Matrix N;                       // Matrix N (2m x 2m)

      Vector rc = Z.transpose()*(g + th*(xc - x) - W*M*c);  // Reduced gradient of the quadratic model mk at xc (tf x 1)

      // Compute v and N using the active set or free set, whichever is smaller
      if (A.cols() < Z.cols()) {
        v = Mb*W.transpose()*Z*rc;

        // If A is empty, N remains empty (mathematically N = I)
        if (A.cols() > 0) {
          N = Matrix::Identity(Mb.rows(), Mb.cols()) + th*Mb*Wb.transpose()*A*A.transpose()*Wb;
        }
      }
      else {
        v = M*W.transpose()*Z*rc;

        // If Z is empty, N remains empty (mathematically N = I)
        if (Z.cols() > 0) {
          N = Matrix::Identity(M.rows(), M.cols()) - th_inv*M*W.transpose()*Z*Z.transpose()*W;
        }
      }

      // Compute v = N^-1*v (2m x 1)
      // If N is empty here, mathematically N = I therefore N^-1*v = v and we can skip this calculation
      if (N.size() > 0) {
        v = N.lu().solve(v);
      }

      // Compute du = -B^-1*rc = -(1/th)*rc - (1/th^2)*Z^T*W*v (tf x 1)
      //
      // Note: There is an error in the paper here. Equation 5.7 correctly specifies the solution to m(d) as
      //       d = -B^-1*rc, then Equation 5.11 mistakenly drops the (-) sign and states d = B^-1*rc. The calculation
      //       below follows the correct solution to m(d), d = -B^-1*rc, which can be verified by setting the derivative
      //       m'(d) = 0 and solving for d.
      Vector du = -th_inv*rc - th_inv*th_inv*Z.transpose()*W*v;

      // Find a_star = max{a : a <= 1, l(i) - xc(i) <= a*du(i) <= u(i) - xc(i), i ∈ F}
      Scalar a_star = std::min<Scalar>(1.0, maxStep(xc(free_set), l(free_set), u(free_set), du));

      // Compute d (n x 1)
      // = xc(i) - x                    if i ∉ F
      // = xc(i) - x + (Z*a_star*du)(i) if i ∈ F
      Vector d = xc - x;
      Vector Zd_star = Z*a_star*du;

      for (Index i : free_set) {
        d(i) += Zd_star(i);
      }

      return d;
    }

    // Performs the limited-memory BFGS update of th, S, Y, W, Wb, M, Mb and related matrices.
    template <class Derived>
    void updateMatrices(const Eigen::MatrixBase<Derived>& s,
                        const Eigen::MatrixBase<Derived>& y
                       )
    {
      // Discard {s, y} if the curvature condition s^T * y > 0 is not satisified
      if (s.dot(y) > ScalarLimits::epsilon()*y.squaredNorm()) {
        // Increase the size m of the matrices with each correction pair we keep up to m_max
        m = std::min(m+1, m_max);

        // Once we've reached the max memory size, begin discarding old {s, y} pairs by shifting matrix data left & up
        if (Y.cols() == m_max) {
          shift(S, 0, -1);
          shift(Y, 0, -1);
          shift(SS, -1, -1);
          shift(SY, -1, -1);
          shift(YY, -1, -1);
        }
        else {
          S.conservativeResize(Eigen::NoChange, m);
          Y.conservativeResize(Eigen::NoChange, m);
          SS.conservativeResize(m, m);
          SY.conservativeResize(m, m);
          YY.conservativeResize(m, m);

          W.conservativeResize(Eigen::NoChange, 2*m);
          Wb.conservativeResize(Eigen::NoChange, 2*m);

          M.conservativeResize(2*m, 2*m);
          Mb.conservativeResize(2*m, 2*m);
        }

        // Update th
        th = y.dot(y)/y.dot(s);
        th_inv = 1/th;

        // Add new s, y to S and Y
        S.col(m-1) = s;
        Y.col(m-1) = y;

        // Update SS
        SS.row(m-1) = S.col(m-1).transpose()*S;
        SS.col(m-1) = SS.row(m-1).transpose();

        // Update SY
        SY.row(m-1) = S.col(m-1).transpose()*Y;
        SY.col(m-1) = S.transpose()*Y.col(m-1);

        // Update YY
        YY.row(m-1) = Y.col(m-1).transpose()*Y;
        YY.col(m-1) = YY.row(m-1).transpose();

        // Form D (diagonal of S^T*Y) and R^-1 (inverse of the upper triangular matrix of S^T*Y)
        D = SY.diagonal().asDiagonal();
        R_inv = SY.template triangularView<Eigen::Upper>();
        R_inv = R_inv.inverse().eval();

        // Update W
        W.leftCols(m) = Y;
        W.rightCols(m) = th*S;

        // Update Wb
        Wb.leftCols(m) = th_inv*Y;
        Wb.rightCols(m) = S;

        // Update M
        M.topLeftCorner(m, m) = -D;
        M.topRightCorner(m, m) = SY.template triangularView<Eigen::StrictlyLower>().transpose();
        M.bottomLeftCorner(m, m) = SY.template triangularView<Eigen::StrictlyLower>();
        M.bottomRightCorner(m, m) = th*SS;
        M = M.inverse().eval();

        // Update Mb
        Mb.topLeftCorner(m, m) = Matrix::Zero(m, m);
        Mb.topRightCorner(m, m) = -R_inv;
        Mb.bottomLeftCorner(m, m) = (-R_inv).transpose();
        Mb.bottomRightCorner(m, m) = R_inv.transpose()*(D + th_inv*YY)*R_inv;
      }
    }

    // Finds max{ t: l(i) <= x(i) + t*d(i) <= u(i) ∀ d(i) }
    template <class Fixed, class Dynamic>
    Scalar maxStep(const Eigen::MatrixBase<Fixed>& x,
                   const Eigen::MatrixBase<Fixed>& l,
                   const Eigen::MatrixBase<Fixed>& u,
                   const Eigen::MatrixBase<Dynamic>& d
                  )
    {
      // Scaler to prevent precision issues from generating a max step that violates the bounds by a small amount
      constexpr Scalar s = 1 - ScalarLimits::epsilon();

      Scalar t = 0.0;
      Scalar t_max = ScalarLimits::max();

      for (Index i = 0; i < d.size(); ++i) {
        if (d(i) != 0.0) {
          t = s * std::max<Scalar>((u(i) - x(i))/d(i),
                                   (l(i) - x(i))/d(i)
                                  );
          t = clip(t, 0.0, ScalarLimits::max());
        }
        else {
          t = ScalarLimits::max();
        }
        t_max = std::min<Scalar>(t, t_max);
      }

      return t_max;
    }

  private:
    const Index m_max = 5;  // Maximum number of correction pairs {s, y} to store in the limited memory matrices

    LineSearch line_search{}; // Line search method

    Index m = 0;         // Current number of correction pairs {s, y} stored in the limited memory matrices
    Scalar th = 1.0;     // Theta scaling parameter
    Scalar th_inv = 1.0; // 1/Theta scaling parameter

    Matrix I{};        // Identity Matrix (n x n)
    Matrix S{};        // Matrix S where each element is the correction pair xk+1 - xk (n x m)
    Matrix Y{};        // Matrix Y where each element is the correction pair gk+1 - gk (n x m)
    Matrix SS{};       // Matrix S^T*S (m x m)
    Matrix SY{};       // Matrix S^T*Y (m x m)
    Matrix YY{};       // Matrix Y^T*Y (m x m)

    Matrix D{};        // Diagonal matrix of S^T*Y (m x m)
    Matrix R_inv{};    // Inverse of the upper triangular matrix of S^T*Y (m x m)

    Matrix W{};        // Matrix W from the L-BFGS-B paper (n x 2m)
    Matrix Wb{};       // Matrix Wb from the L-BFGS-B paper (n x 2m)

    Matrix M{};        // Matrix M from the L-BFGS-B paper (2m x 2m)
    Matrix Mb{};       // Matrix Mb from the L-BFGS-B paper (2m x 2m)

    Vector c{};        // Vector which represents a component needed to compute the reduced gradient of free
                     // variables at the Cauchy point (2m x 1)

    IndexSet free_set{};    // Set of free (unconstrained) indexes (tf x 1)
    IndexSet active_set{};  // Set of active (constrained) indexes (ta x 1)
  };

} // namespace optimize


// Copyright (c) 2026
// Distributed under the terms of the MIT License


#include <cmath>          // isfinite, sqrt


namespace optimize
{
  // One-dimensional, box-constrained minimizer.
  //
  // Minimize1D uses a safeguarded Newton direction when Function::computeHessian()
  // supplies a positive 1x1 Hessian. If the Hessian is unavailable, non-positive, or
  // numerically unusable, it falls back to steepest descent. Every direction is kept
  // inside the box constraints and accepted by a Lewis-Overton strong-Wolfe line search;
  // a weak-Wolfe search is used as a fallback for less regular objectives.
  class Minimize1D final : public Solver
  {
  public:
    using Solver::Callback;
    using Solver::minimize;

    explicit Minimize1D(const Callback& callback = [](Solver*) {}) :
      Solver(callback)
    {}

    // Minimizes f on the interval [lower, upper], starting at x. The bounds are
    // normalized by Solver, so their order is immaterial. The returned State stores
    // its scalar argument, value and gradient in one-element Eigen vectors.
    const State& minimize(Function& f,
                          const Scalar x,
                          const Scalar lower = ScalarLimits::lowest(),
                          const Scalar upper = ScalarLimits::max()
                         )
    {
      return Solver::minimize(f,
                              Vector::Constant(1, x),
                              Vector::Constant(1, lower),
                              Vector::Constant(1, upper)
                             );
    }

  private:
    void performStep(Function& f) override
    {
      Iterate& previous = prev_state.iterate;
      Iterate& current = curr_state.iterate;
      const Scalar gradient = current.g(0);

      // A zero gradient is already stationary. The projected-gradient convergence
      // check in Solver::updateState() handles the equivalent bound-constrained case.
      if (gradient == 0.0) {
        return;
      }

      // Newton is a descent direction only for a strictly positive usable curvature.
      // The default Function implementation returns an empty Hessian, which naturally
      // selects the gradient-descent fallback without requiring a special function type.
      const Matrix hessian = f.hessian(current);
      Scalar direction_value = -gradient;
      if (   hessian.rows() == 1
          && hessian.cols() == 1
          && std::isfinite(hessian(0, 0))
          && hessian(0, 0) > std::sqrt(ScalarLimits::epsilon())
         ) {
        const Scalar newton_direction = -gradient/hessian(0, 0);
        if (std::isfinite(newton_direction) && gradient*newton_direction < 0.0) {
          direction_value = newton_direction;
        }
      }

      Vector direction = Vector::Constant(1, direction_value);
      const Scalar t_max = maxStep(current.x(0), direction_value);
      if (t_max <= 0.0) {
        return;
      }

      Scalar step = strong_line_search(f, current.f, current.x, current.g, direction, t_max);
      if (step == 0.0) {
        step = weak_line_search(f, current.f, current.x, current.g, direction, t_max);
      }
      if (step == 0.0) {
        return;
      }

      previous = current;
      current.x += step*direction;
      current.f = f(current.x);
      current.g = f.gradient(current.x);
    }

    // Returns the largest nonnegative line-search parameter that keeps x + t*d
    // within the current one-dimensional box. An unbounded side yields no limit.
    Scalar maxStep(const Scalar x, const Scalar direction) const
    {
      if (direction > 0.0) {
        return u(0) == ScalarLimits::max()
             ? ScalarLimits::max()
             : std::max<Scalar>(0.0, (u(0) - x)/direction);
      }
      if (direction < 0.0) {
        return l(0) == ScalarLimits::lowest()
             ? ScalarLimits::max()
             : std::max<Scalar>(0.0, (l(0) - x)/direction);
      }
      return 0.0;
    }

    LewisOverton<Wolfe::strong> strong_line_search{};
    LewisOverton<Wolfe::weak> weak_line_search{};
  };

} // namespace optimize



#endif // LBFGSB_HH
