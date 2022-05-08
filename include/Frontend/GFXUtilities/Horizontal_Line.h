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
#ifndef _GHORIZONTALLINE
#define _GHORIZONTALLINE

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Horizontal_Line
{
private:
	double y;

public:
	Horizontal_Line();
	Horizontal_Line(double);
	Horizontal_Line(const Horizontal_Line&);
	~Horizontal_Line();

	// get
	double getY() const;

	// set
	void set(double);
	void setY(double);

	// operators
	inline Horizontal_Line operator+(Horizontal_Line v)
	{
		v.y += y;
		return v;
	}

	inline Horizontal_Line operator-(Horizontal_Line v)
	{
		v.y = y - v.y;
		return v;
	}

	inline Horizontal_Line operator*(double scalar)
	{
		return Horizontal_Line(scalar * y);
	}

	inline Horizontal_Line operator/(double scalar)
	{
		return Horizontal_Line(y / scalar);
	}

	void operator=(const Horizontal_Line& p)
	{
		setY(p.y);
	}

	bool operator==(const Horizontal_Line& v) const
	{
		// y
		double deltaY = v.y - y;
		bool yChange = ((deltaY > -DBL_EPSILON) && (deltaY < DBL_EPSILON));

		return yChange;
	}

	bool operator!=(const Horizontal_Line& v) const
	{
		return !(*this == v);
	}

	bool operator<(const Horizontal_Line& c2) const // ONLY COMPARES X
	{
		return (getY() < c2.getY());
	}

	bool operator>(const Horizontal_Line& c2) const // ONLY COMPARES X
	{
		return (getY() > c2.getY());
	}
};

#endif
