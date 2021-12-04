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
#ifndef _GSCALAR1
#define _GSCALAR1

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Scalar1
{
private:
	double x;

public:
	Scalar1();
	Scalar1(double);
	Scalar1(const Scalar1&);
	~Scalar1();

	// get
	double get() const;
	double length() const;

	// set
	void set(double);

	// operators
	inline Scalar1 operator+(Scalar1 v)
	{
		v.x += x;
		return v;
	}

	inline Scalar1 operator-(Scalar1 v)
	{
		v.x = x - v.x;
		return v;
	}

	inline Scalar1 operator*(double scalar)
	{
		return Scalar1(scalar * x);
	}

	inline Scalar1 operator/(double scalar)
	{
		return Scalar1(x / scalar);
	}

	void operator=(const Scalar1& p)
	{
		set(p.x);
	}

	bool operator==(const Scalar1& v) const
	{
		// x
		double deltaX = v.x - x;
		bool xChange = ((deltaX > -DBL_EPSILON) && (deltaX < DBL_EPSILON));

		return (xChange);
	}

	bool operator!=(const Scalar1& v) const
	{
		return !(*this == v);
	}
};

#endif
