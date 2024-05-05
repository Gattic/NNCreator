#ifndef _DRAWNEURON_H
#define _DRAWNEURON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "Backend/Database/GPointer.h"

class DrawNeuron
{

    protected:
	int edges;
	std::vector<shmea::GPointer<float> > weights;
	float activation;

    public:
	DrawNeuron(int);
	~DrawNeuron();

	float getActivation() const;
	std::vector<shmea::GPointer<float> > getWeights() const;

	void setActivation(float);
	void setWeights(std::vector<shmea::GPointer<float> >);

	void displayNeuron();

};




#endif
