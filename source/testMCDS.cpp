
#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"
#include "common/itkImageProcessingTools.h"
//#include "metric/mcds.h"
//#include "metric/mcds2.h"
//#include "metric/mcds3.h"
//#include "metric/mcds4.h"
//#include "metric/mcds5.h"
//#include "metric/mcds6.h"
//#include "metric/mcds7.h"
#include "metric/mcds8.h"
#include "metric/samplers.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

template <unsigned int Dim>
void DoTest(int argc, char** argv) {
    typedef float ValueType;
    typedef itk::IPT<ValueType, Dim> IPT;

    typedef itk::Image<ValueType, Dim> ImageType;
    typedef typename ImageType::Pointer ImagePointer;

    //typedef MCDS<ImageType, RandomSampler> MCDSType;
    typedef MCDS<ImageType, QMCSampler<1U> > MCDSType;

    RegisterIOFactories();

    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;    

    chronometer.Start("Loading");
    memorymeter.Start("Loading");
    ImagePointer image = IPT::LoadImage(argv[2]);
    chronometer.Stop("Loading");
    memorymeter.Stop("Loading");

    image = IPT::NormalizeImage(image, IPT::IntensityMinMax(image, 0.01));
    std::cout << image << std::endl;

    unsigned int smpls = atoi(argv[3]);
    unsigned int N = argc > 4 ? atoi(argv[4]) : 100000;
    double dmax = 0.0;

    itk::Point<double, Dim> point;
    point[0] = 95.5;
    point[1] = 72.5;//81.5; //70
    //point[0] = 72.5;
    //point[1] = 85.5;
    double h = 0.1;
    if(argc > 5)
        h = atof(argv[5]);
    if(argc > 7 && Dim == 2) {
        point[0] = atof(argv[6]);
        point[1] = atof(argv[7]);
    }
    if(argc > 8 && Dim == 3) {
        point[0] = atof(argv[6]);
        point[1] = atof(argv[7]);
        point[2] = atof(argv[8]);
    }
    if(argc > 8 && Dim == 2)
        dmax = atof(argv[8]);
    if(argc > 9 && Dim == 3)
        dmax = atof(argv[9]);

    unsigned int samples = smpls;
    double value;
    itk::Vector<double, Dim> grad;

    chronometer.Start("Pre-processing");
    memorymeter.Start("Pre-processing");
    MCDSType mcds;

    mcds.SetImage(image);
    mcds.SetOne(1.0);
    mcds.SetMaxDistance(dmax);
    mcds.SetSampleCount(smpls);

    mcds.Initialize();
    chronometer.Stop("Pre-processing");
    memorymeter.Stop("Pre-processing");

    srand(1337);
    chronometer.Start("Evaluation");
    double valsum = 0.0;
    for(unsigned int i = 0; i < N; ++i) {
        itk::Point<double, Dim> curPoint;
        bool any_pos = false;
        for(unsigned int j = 0; j < Dim; ++j) {
            if(point[j] >= 1.0) {
                any_pos = true;
                break;
            }
        }
        if(!any_pos) {
            for(unsigned int j = 0; j < Dim; ++j) {
                curPoint[j] = (rand() / (double)RAND_MAX) * image->GetLargestPossibleRegion().GetSize()[j] * image->GetSpacing()[j];
                //curPoint[j] = 50.0 + 0.5 * (rand() / (double)RAND_MAX) * image->GetLargestPossibleRegion().GetSize()[j] * image->GetSpacing()[j];
            }
        } else {
            curPoint = point;
        }
        
        double hcur = h;
        if(h < 0.0) {
            hcur = rand() / (double)RAND_MAX;
        }
        mcds.ValueAndDerivative(curPoint, hcur, value, grad);
        valsum += value;
        //std::cout << "Value: " << value << ", Grad: " << grad << std::endl;
        //std::cout << grad.GetNorm() << std::endl;
    }
    chronometer.Stop("Evaluation");
    std::cout << "Value: " << value << ", Grad: " << grad << std::endl;
    std::cout << grad.GetNorm() << std::endl;
    std::cout << valsum << std::endl;
    std::cout << "Visited nodes: " << mcds.m_DebugVisitCount/(double)N << std::endl;

    chronometer.Report(std::cout);
    memorymeter.Report(std::cout);
}

int main(int argc, char** argv) {
    int dim = atoi(argv[1]);
    if(dim == 2) {
        DoTest<2U>(argc, argv);
    } else if(dim == 3) {
        DoTest<3U>(argc, argv);
    }
    /*
    typedef float ValueType;
    typedef itk::IPT<ValueType, 2U> IPT;

    typedef itk::Image<ValueType, 2U> ImageType;
    typedef typename ImageType::Pointer ImagePointer;

    //typedef MCDS<ImageType, RandomSampler> MCDSType;
    typedef MCDS<ImageType, QMCSampler<1U> > MCDSType;

    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;    

    chronometer.Start("Loading");
    memorymeter.Start("Loading");
    ImagePointer image = IPT::LoadImage(argv[1]);
    chronometer.Stop("Loading");
    memorymeter.Stop("Loading");

    image = IPT::NormalizeImage(image, IPT::IntensityMinMax(image, 0.01));
    std::cout << image << std::endl;

    unsigned int smpls = atoi(argv[2]);
    unsigned int N = argc > 3 ? atoi(argv[3]) : 100000;
    double dmax = 0.0;

    itk::Point<double, 2U> point;
    point[0] = 95.5;
    point[1] = 72.5;//81.5; //70
    //point[0] = 72.5;
    //point[1] = 85.5;
    double h = 0.1;
    if(argc > 4)
        h = atof(argv[4]);
    if(argc > 6) {
        point[0] = atof(argv[5]);
        point[1] = atof(argv[6]);
    }
    if(argc > 7)
        dmax = atof(argv[7]);

    unsigned int samples = smpls;
    double value;
    itk::Vector<double, 2U> grad;

    chronometer.Start("Pre-processing");
    memorymeter.Start("Pre-processing");
    MCDSType mcds;

    mcds.SetImage(image);
    mcds.SetOne(1.0);
    mcds.SetMaxDistance(dmax);
    mcds.SetSampleCount(smpls);

    mcds.Initialize();
    chronometer.Stop("Pre-processing");
    memorymeter.Stop("Pre-processing");

    srand(1337);
    chronometer.Start("Evaluation");
    double valsum = 0.0;
    for(unsigned int i = 0; i < N; ++i) {
        itk::Point<double, 2U> curPoint;
        if(point[0] < 1.0 && point[1] < 1.0) {
            curPoint[0] = (rand() / (double)RAND_MAX) * image->GetLargestPossibleRegion().GetSize()[0];
            curPoint[1] = (rand() / (double)RAND_MAX) * image->GetLargestPossibleRegion().GetSize()[1];
        } else {
            curPoint = point;
        }
        double hcur = h;
        if(h < 0.0) {
            hcur = rand() / (double)RAND_MAX;
        }
        mcds.ValueAndDerivative(curPoint, hcur, value, grad);
        valsum += value;
        //std::cout << "Value: " << value << ", Grad: " << grad << std::endl;
        //std::cout << grad.GetNorm() << std::endl;
    }
    chronometer.Stop("Evaluation");
    std::cout << "Value: " << value << ", Grad: " << grad << std::endl;
    std::cout << grad.GetNorm() << std::endl;
    std::cout << valsum << std::endl;
    std::cout << "Visited nodes: " << mcds.m_DebugVisitCount/(double)N << std::endl;

    chronometer.Report(std::cout);
    memorymeter.Report(std::cout);*/
    return 0;
}
