#include <cmath>

#include "mpitest.h"

#include "dummy.cpp"

struct int_fixture
{
  static const int n = 8;
  int a[n]           = {1, 2, 3, 4, 5, 6, 7, 8};
  int b[n]           = {0, 1, 2, 3, 4, 5, 6, 7};
};

struct float_fixture
{
  static const int n = 8;
  float a[n]         = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float b[n]         = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
};

TEST(add_test, 2)
{
  int_fixture data;
  arrays<int> state = setup(comm, data.a, data.b, data.n);
  add(&state);
  print(&state);

  switch (state.rank)
  {
    case 0:
      ASSERT_EQ(state.n_local, 3);
      ASSERT_EQ(state.c_local[0], 1);
      break;
    case 1:
      ASSERT_EQ(state.n_local, 4);
      ASSERT_EQ(state.c_local[0], 9);
      break;
  }

  clean(&state);
}

TEST(sub_test, 4)
{
  int_fixture data;
  arrays<int> state = setup(comm, data.a, data.b, data.n);
  sub(&state);
  print(&state);

  switch (state.rank)
  {
    case 0:
      ASSERT_EQ(state.n_local, 2);
      ASSERT_EQ(state.c_local[0], 0);
      break;
    case 1:
      ASSERT_EQ(state.n_local, 2);
      ASSERT_EQ(state.c_local[0], 1);
      break;
    case 2:
      ASSERT_EQ(state.n_local, 2);
      ASSERT_EQ(state.c_local[0], 1);
      break;
    case 3:
      ASSERT_EQ(state.n_local, 2);
      ASSERT_EQ(state.c_local[0], 1);
      break;
  }

  clean(&state);
}

TEST(float_add_test, 2)
{
  float_fixture data;
  arrays<float> state = setup(comm, data.a, data.b, data.n);
  add(&state);
  print(&state);

  switch (state.rank)
  {
    case 0:
      ASSERT_EQ(state.n_local, 4);
      ASSERT_FLOAT_EQ(state.c_local[0], 0.100001, 10);
      break;
    case 1:
      ASSERT_EQ(state.n_local, 4);
      ASSERT_FLOAT_EQ(state.c_local[0], NAN, 10);
      break;
  }

  clean(&state);
}

/* some random serial tests that don't do much */

TEST(serial_add, 1)
{
  ASSERT_EQ(2, 1 + 1);
}

TEST(serial_double, 1)
{
  ASSERT_EQ(1337., 1337.);
}

TEST(serial_float, 1)
{
  ASSERT_FLOAT_EQ(1337.f, 1337.f, 10);
}

TEST(serial_double_zero, 1)
{
  ASSERT_DOUBLE_EQ(0., 1e-8, 10, 5e-8);
}

TEST(serial_double_signed_zero, 1)
{
  ASSERT_DOUBLE_EQ(-0., 1e-6, 10);
}

TEST(serial_double_straddle, 1)
{
  ASSERT_DOUBLE_EQ(-0.000001, 0.000001, 10);
}
