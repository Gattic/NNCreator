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
#ifndef _GPOINT2
#define _GPOINT2

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "Backend/Database/StandItem.h"

class Point2 : public shmea::StandItem
{
private:
	double x, y;

public:
	Point2();
	Point2(double, double);
	Point2(const Point2&);
	virtual ~Point2();

	// get
	double getX() const;
	double getY() const;
	double length() const;

	// set
	void set(double, double);
	void setX(double);
	void setY(double);
	void normalize();
	virtual shmea::GList toXVectorData() const;
	virtual shmea::GList toYVectorData() const;

	// operators
	inline Point2 operator+(Point2 v)
	{
		v.x += x;
		v.y += y;
		return v;
	}

	inline Point2 operator-(Point2 v)
	{
		v.x = x - v.x;
		v.y = y - v.y;
		return v;
	}

	inline Point2 operator*(double scalar)
	{
		return Point2(scalar * x, scalar * y);
	}

	inline Point2 operator/(double scalar)
	{
		return Point2(x / scalar, y / scalar);
	}

	void operator=(const Point2& p)
	{
		setX(p.x);
		setY(p.y);
	}

	bool operator==(const Point2& v) const
	{
		// x
		double deltaX = v.x - x;
		bool xChange = ((deltaX > -DBL_EPSILON) && (deltaX < DBL_EPSILON));
		// y
		double deltaY = v.y - y;
		bool yChange = ((deltaY > -DBL_EPSILON) && (deltaY < DBL_EPSILON));

		return (xChange && yChange);
	}

	bool operator!=(const Point2& v) const
	{
		return !(*this == v);
	}

	bool operator<(const Point2& c2) const // ONLY COMPARES X
	{
		return (getX() < c2.getX());
	}

	bool operator>(const Point2& c2) const // ONLY COMPARES X
	{
		return (getX() > c2.getX());
	}
};

#endif
