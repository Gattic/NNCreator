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
#ifndef _GDATAINPUT
#define _GDATAINPUT

#include "Backend/Database/GTable.h"

namespace glades {

class OHE;

class DataInput
{
public:

	const static int CSV = 0;
	const static int IMAGE = 1;
	const static int TEXT = 2;

	std::vector<OHE*> OHEMaps;
	std::vector<bool> featureIsCategorical;

	virtual void import(shmea::GString) = 0;

	virtual shmea::GList getTrainRow(unsigned int) const = 0;
	virtual shmea::GList getTrainExpectedRow(unsigned int) const = 0;

	virtual shmea::GList getTestRow(unsigned int) const = 0;
	virtual shmea::GList getTestExpectedRow(unsigned int) const = 0;

	virtual unsigned int getTrainSize() const = 0;
	virtual unsigned int getTestSize() const = 0;
	virtual unsigned int getFeatureCount() const = 0;

	virtual int getType() const = 0;
};
};

#endif
