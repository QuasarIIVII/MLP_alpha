project(
	A
	LANGUAGES CXX
	)
add_compile_options(--static --std=c++2a)
add_executable(a a.cpp MLP_alpha/MLP_alpha.h)
add_subdirectory(MLP_alpha)
