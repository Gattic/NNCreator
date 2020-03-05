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
#ifndef _QUATERNION
#define _QUATERNION

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Vec3;

class Quaternion
{
private:
	double w, x, y, z;

public:
	Quaternion();
	Quaternion(double, double, double, double);
	Quaternion(double, Vec3);
	Quaternion(const Quaternion&);
	~Quaternion();

	// get
	double getW() const;
	double getX() const;
	double getY() const;
	double getZ() const;
	double length() const;
	Quaternion getConjugate() const;

	// set
	void set(double, double, double, double);
	void setW(double);
	void setX(double);
	void setY(double);
	void setZ(double);
	void normalize();

	// operator
	Quaternion operator*(const Quaternion B) const
	{
		Quaternion C((w * B.w) - (x * B.x) - (y * B.y) - (z * B.z),
					 (w * B.x) + (x * B.w) - (y * B.z) + (z * B.y),
					 (w * B.y) + (x * B.z) + (y * B.w) - (z * B.x),
					 (w * B.z) - (x * B.y) + (y * B.x) + (z * B.w));

		return C;
	}

	void operator=(const Quaternion& q)
	{
		setW(q.w);
		setX(q.x);
		setY(q.y);
		setZ(q.z);
	}

	bool operator==(const Quaternion q) const
	{
		// w
		double deltaW = q.w - w;
		bool wChange = ((deltaW > -DBL_EPSILON) && (deltaW < DBL_EPSILON));
		// x
		double deltaX = q.x - x;
		bool xChange = ((deltaX > -DBL_EPSILON) && (deltaX < DBL_EPSILON));
		// y
		double deltaY = q.y - y;
		bool yChange = ((deltaY > -DBL_EPSILON) && (deltaY < DBL_EPSILON));
		// z
		double deltaZ = q.z - z;
		bool zChange = ((deltaZ > -DBL_EPSILON) && (deltaZ < DBL_EPSILON));

		return (deltaW && xChange && yChange && zChange);
	}
};

#endif
