#include <cstdio>

#include "mpitest.h"

#define FAIL_MESSAGE_SIZE 1024

namespace mpi_test
{
  test_list* test_list::instance()
  {
    static test_list list;
    return &list;
  }

  test_ptr add_test(test_ptr test,
                    std::initializer_list<int> test_sizes,
                    const char* name)
  {
    for (int test_size : test_sizes)
      test_list::instance()->tests.push_back({test, test_size, name});
    return test;
  }

  void register_error(const assert_info assertion, std::string reason)
  {
    test_list* list = test_list::instance();
    list->tests[list->current_test].fails.push_back({assertion, reason});
  }

}  // namespace mpi_test

int main(int argc, char** argv)
{
  /* initialize and make a little space */

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  mpi_test::test_list* list = mpi_test::test_list::instance();

  if (rank == 0)
    printf("\n\n");

  MPI_Barrier(MPI_COMM_WORLD);

  /* ensure the program was launched with enough procs */

  // Note that all processors generate the same test list during dynamic
  // initialization (at least I think that's how it works). Each processor needs
  // to do the work of finding the largest test so they all hit the
  // MPI_Finalize() in the case of an incorrect launch size (without
  // communication, that is).

  int largest_test_size = 0;
  for (int ti = 0; ti < list->tests.size(); ++ti)
  {
    if (list->tests[ti].test_size > largest_test_size)
      largest_test_size = list->tests[ti].test_size;
  }

  if (size < largest_test_size)
  {
    if (rank == 0)
      printf("please launch with at least %d procs!\n", largest_test_size);

    MPI_Finalize();
    return 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* run each test */

  for (int ti = 0; ti < list->tests.size(); ++ti)
  {
    // This is a reference because all test information resides in the single
    // test_list instance.
    mpi_test::test_info& this_test = list->tests[ti];

    // The error registration routine needs the current test to be set in the
    // (single) test list instance to assign failure information to the correct
    // test, so it's set here.
    // Maybe this should actually be passed through in the future...
    list->current_test = ti;

    // Make communicator for this test.
    // Each test runs using it's own communicator, this is critical so that the
    // test code and mpi_test to not interfere with each other. Communicator
    // splitting is used here, the lowest [proc count] processors are actually
    // used in any given test. The two new communicators are differentiated by
    // using the processor's rank in MPI_COMM_WORLD in this routine.
    int color = 0;
    if (rank < this_test.test_size)
      color = 1;
    MPI_Comm test_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &test_comm);

    // Run this test on the appropriate number of procs (within the test_comm).
    // Other processors fall through and stop at the barrier.
    if (rank < this_test.test_size)
    {
      // start of test print
      if (rank == 0)
      {
        const char* plural = (this_test.test_size > 1) ? "s" : "";
        printf("[ RUNNING ] %s (%d proc%s)\n",
               this_test.test_name,
               this_test.test_size,
               plural);
      }

      // run the test
      this_test.fptr(test_comm);

      // Since all failure information must be printed from a single process (I
      // don't believe print synchronization works across processes even if the
      // processes issue print commands in order due to buffer flushing issues)
      // and you don't have to communicate anything if there's no failure, the
      // number of failures on each process is communicated back in advance
      // here.
      int* num_fails_by_rank;
      if (rank == 0)
        num_fails_by_rank = new int[this_test.test_size];

      int num_fails_local = (int)this_test.fails.size();
      MPI_Gather(&num_fails_local,
                 1,
                 MPI_INT,
                 num_fails_by_rank,
                 1,
                 MPI_INT,
                 0,
                 test_comm);

      // give a success print if there were no failures
      int num_fails_total = 0;
      if (rank == 0)
      {
        for (int ri = 0; ri < this_test.test_size; ++ri)
          num_fails_total += num_fails_by_rank[ri];

        if (num_fails_total == 0)
          printf("[ SUCCESS ] %s\n", this_test.test_name);
      }

      // Note that in this next section nothing will be sent, received, or
      // printed if there are no failures, things will just fall through to the
      // next barrier. There might be a better way to do this hmm.

      // Compile an error message locally and send to the root process if there
      // were failures. Failure message numbers are stored in the tag so there's
      // no confusion with the messages.
      MPI_Request* requests = new MPI_Request[this_test.fails.size()];
      MPI_Status* statuses  = new MPI_Status[this_test.fails.size()];
      char* error_message_send_buff =
      new char[FAIL_MESSAGE_SIZE * this_test.fails.size()];
      for (int fi = 0; fi < this_test.fails.size(); ++fi)
      {
        snprintf(error_message_send_buff + (FAIL_MESSAGE_SIZE * fi),
                 FAIL_MESSAGE_SIZE,
                 "  %s FAILED (on proc %d line %d of %s)\n    %s",
                 this_test.fails[fi].assertion.test_string,
                 rank,
                 this_test.fails[fi].assertion.line,
                 this_test.fails[fi].assertion.file,
                 this_test.fails[fi].reason.c_str());

        MPI_Issend(error_message_send_buff + (FAIL_MESSAGE_SIZE * fi),
                   FAIL_MESSAGE_SIZE,
                   MPI_CHAR,
                   0,
                   fi,
                   test_comm,
                   &requests[fi]);
      }

      // On the root process, recieve and print all error messages in order.
      if (rank == 0)
      {
        for (int ri = 0; ri < this_test.test_size; ++ri)
        {
          for (int fi = 0; fi < num_fails_by_rank[ri]; ++fi)
          {
            char error_message_recv[FAIL_MESSAGE_SIZE];

            MPI_Status status;
            MPI_Recv(error_message_recv,
                     FAIL_MESSAGE_SIZE,
                     MPI_CHAR,
                     ri,
                     fi,
                     test_comm,
                     &status);
            printf("%s\n", error_message_recv);
          }
        }
      }

      // Ensure all of your messages are received before freeing the send
      // buffer, the send buffer needs to stick around until the communication
      // completes.
      MPI_Waitall(this_test.fails.size(), requests, statuses);
      delete[] requests;
      delete[] statuses;
      delete[] error_message_send_buff;
      if (rank == 0)
        delete[] num_fails_by_rank;

      // end of test print
      if (rank == 0 && num_fails_total != 0)
        printf("[ FAIL    ] %s\n", this_test.test_name);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_free(&test_comm);
  }

  /* now we clean */

  MPI_Finalize();
  return 0;
}
