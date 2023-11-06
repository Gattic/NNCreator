// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _GQL_LAYERINFO
#define _GQL_LAYERINFO

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class GList;
};

namespace glades {

class LayerInfo
{
protected:
	int lSize;
	float learningRate;
	float momentumFactor;
	float weightDecay1;
	float weightDecay2;
	float pDropout;
	int activationType;
	float activationParam;

public:
	static const int INPUT = 0;
	static const int HIDDEN = 1;
	static const int OUTPUT = 2;

	LayerInfo(int);
	virtual ~LayerInfo();

	void copyParamsFrom(const LayerInfo*);

	// gets
	unsigned int size() const;
	float getLearningRate() const;
	float getMomentumFactor() const;
	float getWeightDecay1() const;
	float getWeightDecay2() const;
	float getPDropout() const;
	int getActivationType() const;
	float getActivationParam() const;
	virtual shmea::GList getGTableRow() const = 0;

	// sets
	void setSize(int);
	void setLearningRate(float);
	void setMomentumFactor(float);
	void setWeightDecay1(float);
	void setWeightDecay2(float);
	void setPDropout(float);
	void setActivationType(int);
	void setActivationParam(float);

	// type
	virtual int getLayerType() const = 0;
};
};

#endif
