CXXFLAGS := -std=c++11
CPPFLAGS := -MMD -MP -Impitest

include local.mk

.PHONY: library
library: lib/libmpitest.a

.PHONY: dummy
dummy: bin/dummy_tests

# main library

lib/libmpitest.a: build/mpitest.o
	@mkdir -p lib
	ar rcs $@ $^

build/mpitest.o: mpitest/mpitest.cpp
	@mkdir -p build
	mpic++ -c ${CPPFLAGS} ${CXXFLAGS} -o$@ mpitest/mpitest.cpp

-include build/mpitest.d

# small test code

bin/dummy_tests: lib/libmpitest.a build/dummy_tests.o
	@mkdir -p bin
	mpic++ ${CPPFLAGS} ${CXXFLAGS} -o$@ build/dummy_tests.o -Llib -lmpitest

build/dummy_tests.o: dummy/dummy_tests.cpp
	@mkdir -p build
	mpic++ -c ${CPPFLAGS} ${CXXFLAGS} -o$@ dummy/dummy_tests.cpp

-include build/dummy_tests.d

# cleanup

.PHONY: clean
clean:
	rm -rf build bin lib
