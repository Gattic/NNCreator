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
#ifndef _GRAPHSCATTER_H
#define _GRAPHSCATTER_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../GItems/RUColors.h"
#include "Graphable.h"

class gfxpp;
class RUGraph;
class Point2;

class GraphScatter : public Graphable
{
private:
	void drawPoint(gfxpp*, int, int, int = 0);
	void drawPointOutline(gfxpp*, int, int, int = 0);
	int pointSize;

public:
	// constructors & destructor
	GraphScatter(RUGraph*, SDL_Color = RUColors::DEFAULT_COLOR_LINE, int = 4);
	~GraphScatter();

	void setPointSize(int);
	int getPointSize(int);

	virtual void draw(gfxpp*);
	virtual std::string getType() const;
};

#endif
