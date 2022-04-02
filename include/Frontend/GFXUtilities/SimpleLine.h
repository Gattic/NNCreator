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
#ifndef _GSIMPLELINE
#define _GSIMPLELINE

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class SimpleLine
{
private:
	double y;

public:
	SimpleLine();
	SimpleLine(double);
	SimpleLine(const SimpleLine&);
	~SimpleLine();

	// get
	double getY() const;

	// set
	void set(double);
	void setY(double);

	// operators
	inline SimpleLine operator+(SimpleLine v)
	{
		v.y += y;
		return v;
	}

	inline SimpleLine operator-(SimpleLine v)
	{
		v.y = y - v.y;
		return v;
	}

	inline SimpleLine operator*(double scalar)
	{
		return SimpleLine(scalar * y);
	}

	inline SimpleLine operator/(double scalar)
	{
		return SimpleLine(y / scalar);
	}

	void operator=(const SimpleLine& p)
	{
		setY(p.y);
	}

	bool operator==(const SimpleLine& v) const
	{
		// y
		double deltaY = v.y - y;
		bool yChange = ((deltaY > -DBL_EPSILON) && (deltaY < DBL_EPSILON));

		return yChange;
	}

	bool operator!=(const SimpleLine& v) const
	{
		return !(*this == v);
	}

	bool operator<(const SimpleLine& c2) const // ONLY COMPARES X
	{
		return (getY() < c2.getY());
	}

	bool operator>(const SimpleLine& c2) const // ONLY COMPARES X
	{
		return (getY() > c2.getY());
	}
};

#endif
