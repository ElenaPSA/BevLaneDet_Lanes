
#include "pybind11/pybind11.h"
#include <pybind11/stl_bind.h>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "ndarray_converter.h"

namespace py = pybind11;


void loadTrackerMedianFlow(py::module &);
void loadTrackSmoother(py::module&);

void printTest()
{
    std::cout << "Ceci est un test" << std::endl;
}


PYBIND11_MODULE(TrackingUtilsPy, m) {

    // To use _a for argument declarations
    using namespace pybind11::literals;

    NDArrayConverter::init_numpy();

    m.doc() = R"pbdoc(
         Python Package
        --------------------------
    )pbdoc";

    // Common objects =========================================================
   

    m.def("printTest", &printTest,
        "Get position from a pose matrix");
   
    // Load modules  ==========================================================

    loadTrackSmoother(m);
   
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "1.0.0";
#endif
}
