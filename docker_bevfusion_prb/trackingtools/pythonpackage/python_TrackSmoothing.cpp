

#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>

#include "pybind11/stl.h"  //for vector conversions
#include <pybind11/stl_bind.h>
#include <vector>
#include <opencv2/opencv.hpp>
PYBIND11_MAKE_OPAQUE(std::vector<cv::Point2f>);
PYBIND11_MAKE_OPAQUE(std::vector<cv::Point3f>);
PYBIND11_MAKE_OPAQUE(std::vector<cv::KeyPoint>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);


#include "IMMFilters.hpp"
#include "TypesDefinitions.hpp"
#include "ndarray_converter.h"

namespace py = pybind11;



class TrackSmoothingPy {
public:
    TrackSmoothingPy(py::array_t<double> init_data)
                     
    {
        //center x, center y, center z,orientation,l,w,h,
      
        py::buffer_info info = init_data.request();
      
        auto buffer = static_cast<double*>(info.ptr);
        
        m_filter.reset(new CVIMMFilterBev(buffer[0],buffer[1],buffer[2],buffer[3],buffer[5],buffer[6],buffer[4]));
      
    };
    int getNumFilters()
    {
        return m_filter->getNumFilters();
    }
    void predict(double dt)
    {
        m_filter->predict(dt);
    }
     void predictInEgoFrame(double dt,double dx, double dy, double dyaw)
    {
        //std::cout<<" predict in ego frame "<<dx<<" "<<dy<<" "<<dyaw<<std::endl;
        m_filter->predictInEgoFrame(dt,dx,dy,dyaw);
    }

    void update(py::array_t<double> obs_data)
    {
        py::buffer_info info = obs_data.request();
      
        auto buffer = static_cast<double*>(info.ptr);
        std::vector<double> obs(7);
        obs[0] = buffer[0];
        obs[1] = buffer[1];
        obs[2] = buffer[2];
        obs[3] = buffer[3]; 
        obs[4] = buffer[5];
        obs[5] = buffer[6];
        obs[6] = buffer[4];
        
        m_filter->correct(obs);
    }

    py::array_t<double> getState()
    {
        TrackOutput track;

       
        py::array_t<double> track_array(9);
        py::buffer_info info = track_array.request();
      
        auto tracks_buffer = static_cast<double*>(info.ptr);
        
        m_filter->getEstimation(track);
    
        tracks_buffer[0] = track.x;
        tracks_buffer[1] = track.y;
        tracks_buffer[2] = track.z;
        tracks_buffer[3] = track.theta
        ;
        tracks_buffer[4] = track.length;
        tracks_buffer[5] = track.width;
        tracks_buffer[6] = track.height;

        tracks_buffer[7] = track.V;
        tracks_buffer[8] = track.yawRate;

        return track_array;
    }

    cv::Mat getFilterState()
    {
        const cv::Mat& state=m_filter->state();
        return state.clone();
    }
    cv::Mat getFilterCovariance()
    {
        const cv::Mat& covar=m_filter->covar();
        return covar.clone();
    }
    cv::Mat getSubFilterState(int i)
    {
        auto filters=m_filter->getFilters();
        return filters[i]->m_state.clone();
    }
    cv::Mat getMixState(int i)
    {
        auto state=m_filter->getMixedStates()[i];
        return state.clone();
    }
    cv::Mat getMixCov(int i)
    {
        auto cov=m_filter->getMixedCov()[i];
        return cov.clone();
    }
    cv::Mat getSubFilterCovariance(int i)
    {
        auto filters=m_filter->getFilters();
        return filters[i]->m_errorCov.clone();
    }
    cv::Mat getSubFilterCrossCovariance(int i)
    {
        auto filters=m_filter->getFilters();
        return filters[i]->m_errorCrossCov.clone();
    }
   
    void predictSubFilter(cv::Mat state, cv::Mat covar,int i,double dt,double dx)
    {
        auto filters=m_filter->getFilters();
        state.copyTo(filters[i]->m_state);
		covar.copyTo(filters[i]->m_errorCov);
        
        bool ego_static = (dx==0.0);
       
        filters[i]->predict(dt,ego_static);
    }

    cv::Mat getTransitionMatrix()
    {
        return m_filter->transitionMatrix().clone();
    }

    py::array_t<double> getModelProbabilities()
    {
        cv::Mat probs= m_filter->getModelProbabilities();

        py::array_t<double> track_array(m_filter->getNumFilters());
        py::buffer_info info = track_array.request();
      
        auto tracks_buffer = static_cast<double*>(info.ptr);

        for (int i=0;i<m_filter->getNumFilters();i++)
        {
            tracks_buffer[i]=probs.at<double>(i);
        }
        return track_array;
    }
    
    std::shared_ptr<CVIMMFilterBev> m_filter;
   
};


void loadTrackSmoother(py::module &m)
{
    // To use _a for argument declarations
    using namespace pybind11::literals;
  
    // Main class =============================================================
    py::class_<TrackSmoothingPy>(m, "Filter")
        .def(py::init([](py::array_t<double> obs_init)
         {
                 return std::unique_ptr<TrackSmoothingPy>(new TrackSmoothingPy(obs_init));
            }))
        .def("predict", &TrackSmoothingPy::predict,"delta_t"_a)
        .def("predictInEgoFrame", &TrackSmoothingPy::predictInEgoFrame,"delta_t"_a,"ego_dx"_a,"ego_dy"_a,"ego_dyaw"_a)
        .def("update", &TrackSmoothingPy::update,"buffer"_a)
        .def("getState", &TrackSmoothingPy::getState)
        .def("getModelProbabilities", &TrackSmoothingPy::getModelProbabilities)
        .def("getFilterState", &TrackSmoothingPy::getFilterState)
        .def("getFilterCovariance", &TrackSmoothingPy::getFilterCovariance)
        .def("getNumFilters", &TrackSmoothingPy::getNumFilters)
        .def("getSubFilterState", &TrackSmoothingPy::getSubFilterState,"index"_a)
        .def("getMixState", &TrackSmoothingPy::getMixState,"index"_a)
        .def("getMixCov", &TrackSmoothingPy::getMixCov,"index"_a)
        .def("getSubFilterCovariance", &TrackSmoothingPy::getSubFilterCovariance,"index"_a)
        .def("getSubFilterCrossCovariance", &TrackSmoothingPy::getSubFilterCrossCovariance,"index"_a)
        .def("getTransitionMatrix", &TrackSmoothingPy::getTransitionMatrix)
        .def("predictSubFilter", &TrackSmoothingPy::predictSubFilter,"state"_a,"covar"_a,"index"_a,"delta_t"_a,"dx"_a);
}