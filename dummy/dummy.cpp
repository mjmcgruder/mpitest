#ifndef DUMMY
#define DUMMY

#include <iostream>

#include <mpi.h>

template<typename real>
struct arrays
{
  int rank;
  int size;
  MPI_Comm comm;
  MPI_Datatype type;
  int n_local;
  real* a_local;
  real* b_local;
  real* c_local;
};

arrays<float> init(float foo)
{
  arrays<float> arr;
  arr.type = MPI_FLOAT;
  return arr;
}

arrays<double> init(double foo)
{
  arrays<double> arr;
  arr.type = MPI_DOUBLE;
  return arr;
}

arrays<int> init(int foo)
{
  arrays<int> arr;
  arr.type = MPI_INT;
  return arr;
}

template<typename real>
arrays<real> setup(MPI_Comm comm, real* a, real* b, int n)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  arrays<real> state = init((real)42);

  /* now we need to split the arrays */

  if (n % size != 0 && rank == 0)
  {
    printf("choose an array size that divides evenly!");
    MPI_Abort(comm, 1);
  }
  MPI_Barrier(comm);

  state.comm    = comm;
  state.rank    = rank;
  state.size    = size;
  state.n_local = n / size;
  state.a_local = new real[state.n_local];
  state.b_local = new real[state.n_local];
  state.c_local = new real[state.n_local];

  int offset = rank * state.n_local;
  for (int i = 0; i < state.n_local; ++i)
  {
    state.a_local[i] = a[i + offset];
    state.b_local[i] = b[i + offset];
  }

  return state;
}

template<typename real>
void clean(arrays<real>* state)
{
  delete[] state->a_local;
  delete[] state->b_local;
  delete[] state->c_local;
}

template<typename real>
void print(arrays<real>* state)
{
  /* now we output (it's brain time) */

  MPI_Request request;
  MPI_Issend(state->c_local,
             state->n_local,
             state->type,
             0,
             state->rank,
             state->comm,
             &request);

  if (state->rank == 0)
  {
    int n   = state->n_local * state->size;
    real* c = new real[n];
    for (int r = 0; r < state->size; ++r)  // receiving from each rank
    {
      MPI_Status status;
      MPI_Recv(c + state->n_local * r,
               state->n_local,
               state->type,
               r,
               r,
               state->comm,
               &status);
    }
    for (int i = 0; i < n; ++i)
      std::cout << c[i] << " ";
    std::cout << "\n";

    delete[] c;
  }

  MPI_Status status;
  MPI_Wait(&request, &status);
}

template<typename real>
void add(arrays<real>* state)
{
  for (int i = 0; i < state->n_local; ++i)
    state->c_local[i] = state->a_local[i] + state->b_local[i];
}

template<typename real>
void sub(arrays<real>* state)
{
  for (int i = 0; i < state->n_local; ++i)
    state->c_local[i] = state->a_local[i] - state->b_local[i];
}

#endif
