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
#ifndef _VEC3
#define _VEC3

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Vec3
{
private:
	double x, y, z;

public:
	Vec3();
	Vec3(double, double, double);
	Vec3(const Vec3&);
	~Vec3();

	// get
	double getX() const;
	double getY() const;
	double getZ() const;
	double length() const;

	// set
	void set(double, double, double);
	void setX(double newX);
	void setY(double newY);
	void setZ(double newZ);
	void normalize();

	// operators
	inline Vec3 operator+(Vec3 v)
	{
		v.x += x;
		v.y += y;
		v.z += z;
		return v;
	}

	inline Vec3 operator-(Vec3 v)
	{
		v.x = x - v.x;
		v.y = y - v.y;
		v.z = z - v.z;
		return v;
	}

	inline Vec3 operator*(double scalar)
	{
		return Vec3(scalar * x, scalar * y, scalar * z);
	}

	inline Vec3 operator/(double scalar)
	{
		return Vec3(x / scalar, y / scalar, z / scalar);
	}

	void operator=(const Vec3& v)
	{
		setX(v.x);
		setY(v.y);
		setZ(v.z);
	}

	bool operator==(const Vec3& v) const
	{
		// x
		double deltaX = v.x - x;
		bool xChange = ((deltaX > -DBL_EPSILON) && (deltaX < DBL_EPSILON));
		// y
		double deltaY = v.y - y;
		bool yChange = ((deltaY > -DBL_EPSILON) && (deltaY < DBL_EPSILON));
		// z
		double deltaZ = v.z - z;
		bool zChange = ((deltaZ > -DBL_EPSILON) && (deltaZ < DBL_EPSILON));

		return (xChange && yChange && zChange);
	}
};

#endif
