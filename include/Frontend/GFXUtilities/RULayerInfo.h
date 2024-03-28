#ifndef _RULAYERINFO
#define _RULAYERINFO

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "Backend/Database/GPointer.h"
#include "DrawNeuron.h"

class RULayerInfo
{
protected:
    int layerType;
    int neuronQty;
//    double* weights;
    std::vector<shmea::GPointer<DrawNeuron> > neuronList;

public:
    static const int INPUT = 0;
    static const int HIDDEN = 1;
    static const int OUTPUT = 2;

    RULayerInfo(int newLayerType, int newNeuronQty, int newPrevLayer=0);
    ~RULayerInfo();
 
    int getNeurons() const;
    std::vector<shmea::GPointer<DrawNeuron> > getNeuronList() const;

    void setNeurons(int newNeuronSize, int newPrevLayer=0);
    void setActivation(int, float);
    void setWeights(int, std::vector<shmea::GPointer<float> >);
    void displayLayer();
    

};



#endif // _RULAYERINFO
