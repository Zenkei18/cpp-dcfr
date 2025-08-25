#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <Python.h>
#include <spdlog/spdlog.h>

// Initialize Python interpreter at the start and finalize it at the end
struct PythonInterpreter {
    PythonInterpreter() {
        spdlog::info("Initializing Python interpreter");
        Py_Initialize();
        
        // Add current directory to Python path
        PyRun_SimpleString("import sys; sys.path.append('.')");
        
        // Import our utility module
        PyRun_SimpleString("import python_utils");
    }
    
    ~PythonInterpreter() {
        spdlog::info("Finalizing Python interpreter");
        Py_Finalize();
    }
};

// Create a single instance of the Python interpreter
PythonInterpreter python_interpreter;
