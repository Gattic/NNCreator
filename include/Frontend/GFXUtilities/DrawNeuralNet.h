#ifndef _DRAWNEURALNET
#define _DRAWNEURALNET

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "RULayerInfo.h"
#include <vector>
#include "Backend/Database/GPointer.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GType.h"
class gfxpp;

class DrawNeuralNet
{

protected:
    int layers;
    std::vector<shmea::GPointer<RULayerInfo> > layerInfoList;
    
    int layerSpacing;
    int neuronSpacing;
    int neuronRadius;

public:
    DrawNeuralNet();    
    DrawNeuralNet(int);
    virtual ~DrawNeuralNet();

    void setInputLayer(int);
    void setHiddenLayer(int, int);
    void setOutputLayer(int);

    void setActivation(shmea::GList);
    void setWeights(shmea::GList);

    int getLayersCount();
    std::vector<shmea::GPointer<DrawNeuron> > getLayerNeurons(int);
    int getLayerNeuronsCount(int);



    void displayNeuralNet();




};




#endif
