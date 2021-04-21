%module halite_swig

%include "std_vector.i"
%begin %{
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

%exception
{
    try
    {
        $action
    }
    catch (const std::exception& e)
    {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%template(float_vector) std::vector<float>;
%{
#define SWIG_FILE_WITH_INIT
#include <Python.h>
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include "board.hpp"
#include "board_store.hpp"
#include "board_config.hpp"
#include "point.hpp"
#include "ship.hpp"
%}

%include "board_config.hpp"
%include "board.hpp"
%include "board_store.hpp"
%include "point.hpp"
%include "ship.hpp"

%template(board_t) Board<BoardConfig>;
%template(board_store_t) BoardStore<BoardConfig>;
