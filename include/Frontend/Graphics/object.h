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
#ifndef _OBJECT
#define _OBJECT

#include "../GFXUtilities/quaternion.h"
#include "../GFXUtilities/vec3.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Object
{
private:
	Vec3 center;
	Vec3 dimensions;
	Quaternion rotation;

	int type;

public:
	static const int EMPTY_TYPE = 0;
	static const int BASIS_TYPE = 1;
	static const int CUBE_TYPE = 2;

	Object();
	Object(const Vec3&, const Vec3&, const Quaternion&);

	void Render();

	// gets
	Vec3 getCenter() const;
	Vec3 getDimensions() const;
	Quaternion getRotation() const;
	int getType() const;

	// sets
	void setCenter(const Vec3&);
	void setDimensions(const Vec3&);
	void setRotation(const Quaternion&);
	void setType(int);

	bool operator==(const Object obj) const
	{
		if (getType() != obj.getType())
			return false;

		else if (getType() == EMPTY_TYPE)
		{
			return true;
		}
		else if (getType() == BASIS_TYPE)
		{
			return true;
		}
		else if (getType() == CUBE_TYPE)
		{
			// check center, dimensions, and rotation
		}

		return false;
	}
};

#endif
