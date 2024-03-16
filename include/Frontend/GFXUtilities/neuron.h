#ifndef _NN_NEURON_H
#define _NN_NEURON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "Backend/Database/GPointer.h"

class Neuron
{

    protected:
	int edges;
	std::vector<shmea::GPointer<float> > weights;
	float activation;

    public:
	Neuron(int);
	~Neuron();

	float getActivation() const;
	std::vector<shmea::GPointer<float> > getWeights() const;

	void setActivation(float);
	void setWeights(std::vector<shmea::GPointer<float> >);

	void displayNeuron();

};




#endif
