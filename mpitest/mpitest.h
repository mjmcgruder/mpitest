#ifndef MPI_TEST
#define MPI_TEST

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

namespace mpi_test
{
  // Function pointer type and signature for all tests.
  typedef void (*test_ptr)(MPI_Comm);

  // Stores information about a particular assertion known at compile time.
  struct assert_info
  {
    const int line;
    const char* file;
    const char* test_string;
  };

  // Stores information about a test failure.
  // Information about the assertion invocation is stored in "assertiion" while
  // the reason for failure is stored in "reason".
  struct fail_info
  {
    const assert_info assertion;
    std::string reason;
  };

  // Stores information about a single test.
  // All test information is populated during dynamic initialization except the
  // "fails" list which will be populated when a test failure occurs.
  struct test_info
  {
    test_ptr fptr;
    int test_size;
    const char* test_name;
    std::vector<fail_info> fails;
  };

  // Stores the list of tests.
  // There should be only a single instance of this struct on each processor
  // accessed through the "instance()" method. The "current_test" should be
  // remain up to date in the main driver code to support error registration at
  // the appropriate test number. The list of tests is build at run time but
  // before "main" is called during dynamic initialization. The "TEST" macro
  // adds to the test list as a byproduct of initializing a random unused
  // variable, so building the test list is really just a byproduct of dynamic
  // initialization of random variables that don't matter but I guess we have to
  // do things this way in C++...
  struct test_list
  {
    int current_test;
    std::vector<test_info> tests;

    static test_list* instance();
  };

  // Adds "test_info" to the single (per processor) "test_list" during dynamic
  // initialization. This function is invoked by the "TEST" macro.
  test_ptr add_test(test_ptr test,
                    std::initializer_list<int> test_sizes,
                    const char* name);

  // The error registration process is always the same, just some compile time
  // information abou the assertion and an exit reason, and the templates need
  // this declared here so ya.
  void register_error(const assert_info assertion, std::string reason);

  /* template assertion implementations */

  // Tests if the given statement is truthy.
  template<typename T>
  bool assert_true(T statement, const assert_info assertion)
  {
    if (statement)
      return true;

    std::stringstream reason;
    reason << statement << " is falsy";
    register_error(assertion, reason.str());
    return false;
  }

  // It seems most simple comparisons (really only thinking of primitives here)
  // can be made with operator==(), so this is the main assertion.
  template<typename T>
  bool assert_eq(T a, T b, const assert_info assertion)
  {
    if (a == b)
      return true;

    std::stringstream reason;
    reason << a << " does not equal " << b;
    register_error(assertion, reason.str());
    return false;
  }

  // Floating point comparisons will not work in general so they have their own
  // logic here. In most cases this function determines if the number of
  // representable floating point numbers between the two given floating point
  // numbers is within the given tolerance "ulp_tol". The exception is if the
  // numbers straddle zero or the numbers are close (absolute values less than
  // "abs_tol") to zero. In this case an absolute comparison is used with a
  // tolerance "abs_tol".
  // See: Comparing Floating Point Numbers, 2012 Edition from Random ASCII for
  // full details.
  template<typename fxx, typename uxx>
  bool assert_ieee754_eq(const assert_info assertion,
                         fxx a,
                         fxx b,
                         int ulp_tol,
                         fxx abs_tol = std::numeric_limits<fxx>::epsilon())
  {
    std::stringstream reason;

    /* quick out if the input is garbage */

    bool quick_exit = false;

    if (std::isnan(a))
    {
      reason << "the first argument is nan! ";
      quick_exit = true;
    }
    else if (std::isinf(a))
    {
      reason << "the first argument is inf! ";
      quick_exit = true;
    }

    if (std::isnan(b))
    {
      reason << "the second argument is nan! ";
      quick_exit = true;
    }
    else if (std::isinf(b))
    {
      reason << "the second argument is inf! ";
      quick_exit = true;
    }

    if (quick_exit)
    {
      register_error(assertion, reason.str());
      return false;
    }

    // reinterpret to unsigned integers

    uxx au = *reinterpret_cast<uxx*>(&a);
    uxx bu = *reinterpret_cast<uxx*>(&b);

    /* absolute comparison */

    if ((std::signbit(a) != std::signbit(b)) ||
        (fabs(a) < abs_tol && fabs(b) < abs_tol))
    {
      fxx diff, tol;
      if ((diff = fabs(a - b)) > (tol = abs_tol))
      {
        reason << "absolute difference between "
               << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << a
               << " and "
               << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << b
               << " ("
               << std::setprecision(std::numeric_limits<fxx>::digits10 + 1)
               << diff << ")"
               << " is outside the requested tolerance " << abs_tol;
        register_error(assertion, reason.str());
        return false;
      }
      return true;
    }

    /* ulp comparison */

    if (au > bu)
    {
      if (au - bu > ulp_tol)
      {
        reason << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << a
               << " and "
               << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << b
               << " differ by " << au - bu
               << " ULPs, the requested tolerance is " << ulp_tol << " ULPs";
        register_error(assertion, reason.str());
        return false;
      }
    }
    else if (bu > au)
    {
      if (bu - au > ulp_tol)
      {
        reason << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << a
               << " and "
               << std::setprecision(std::numeric_limits<fxx>::digits10 + 1) << b
               << " differ by " << bu - au
               << " ULPs, the requested tolerance is " << ulp_tol << " ULPs";
        register_error(assertion, reason.str());
        return false;
      }
    }

    return true;
  }

}  // namespace mpi_test

/* test definition macro */

#define TEST(name, ...)                              \
  void name(MPI_Comm comm);                          \
  mpi_test::test_ptr test_##name =                   \
  mpi_test::add_test(&(name), {__VA_ARGS__}, #name); \
  void name(MPI_Comm comm)

/* assertions */

// Checks if the given statement is truthy.

#define EXPECT_TRUE(a) \
  mpi_test::assert_true((a), {__LINE__, __FILE__, "EXPECT_TRUE(" #a ")"})

#define ASSERT_TRUE(a)                                                     \
  if (!mpi_test::assert_true((a),                                          \
                             {__LINE__, __FILE__, "EXPECT_TRUE(" #a ")"})) \
  {                                                                        \
    return;                                                                \
  }

// Compares two objects with operator==(). They'll also need to support
// operator<<() for error printing but this is only really intended for
// primitives anyway.

#define EXPECT_EQ(a, b) \
  mpi_test::assert_eq(  \
  (a), (b), {__LINE__, __FILE__, "EXPECT_EQ(" #a ", " #b ")"})

#define ASSERT_EQ(a, b)                                             \
  if (!mpi_test::assert_eq(                                         \
      (a), (b), {__LINE__, __FILE__, "ASSERT_EQ(" #a ", " #b ")"})) \
  {                                                                 \
    return;                                                         \
  }

// Note that these macros are variadic, the first variadic argument must be the
// ULP comparison tolerance, it's actually required by both the underlying
// function and the variadic macro becasue variadic macros are weird. The second
// variadic argument is an absolute tolerance for when things are close to zero,
// this one is truly optional. Don't mess up the order!

#define EXPECT_FLOAT_EQ(a, b, ...)                         \
  mpi_test::assert_ieee754_eq<float, uint32_t>(            \
  {__LINE__, __FILE__, "EXPECT_FLOAT_EQ(" #a ", " #b ")"}, \
  (a),                                                     \
  (b),                                                     \
  __VA_ARGS__)

#define ASSERT_FLOAT_EQ(a, b, ...)                             \
  if (!mpi_test::assert_ieee754_eq<float, uint32_t>(           \
      {__LINE__, __FILE__, "ASSERT_FLOAT_EQ(" #a ", " #b ")"}, \
      (a),                                                     \
      (b),                                                     \
      __VA_ARGS__))                                            \
  {                                                            \
    return;                                                    \
  }

#define EXPECT_DOUBLE_EQ(a, b, ...)                         \
  mpi_test::assert_ieee754_eq<double, uint64_t>(            \
  {__LINE__, __FILE__, "EXPECT_DOUBLE_EQ(" #a ", " #b ")"}, \
  (a),                                                      \
  (b),                                                      \
  __VA_ARGS__)

#define ASSERT_DOUBLE_EQ(a, b, ...)                             \
  if (!mpi_test::assert_ieee754_eq<double, uint64_t>(           \
      {__LINE__, __FILE__, "ASSERT_DOUBLE_EQ(" #a ", " #b ")"}, \
      (a),                                                      \
      (b),                                                      \
      __VA_ARGS__))                                             \
  {                                                             \
    return;                                                     \
  }

#endif
