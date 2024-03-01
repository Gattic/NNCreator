#ifndef _RUNEURALNET
#define _RUNEURALNET

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "RULayerInfo.h"
#include <vector>
#include "Backend/Database/GPointer.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GType.h"
class gfxpp;

class NeuralNet
{

protected:
    int layers;
    std::vector<shmea::GPointer<RULayerInfo> > layerInfoList;
    
    int layerSpacing;
    int neuronSpacing;
    int neuronRadius;

public:
    NeuralNet();    
    NeuralNet(int);
    virtual ~NeuralNet();

    void setInputLayer(int);
    void setHiddenLayer(int, int);
    void setOutputLayer(int);

    void setActivation(shmea::GList);
    void setInputLayerActivation();
    void setWeights(shmea::GList);

    int getLayersCount();
    std::vector<shmea::GPointer<Neuron> > getLayerNeurons(int);
    int getLayerNeuronsCount(int);



    void displayNeuralNet();




};




#endif
