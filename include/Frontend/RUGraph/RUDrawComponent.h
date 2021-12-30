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
#ifndef _RUDRAWCOMP
#define _RUDRAWCOMP

#include "RUGraph.h"
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class gfxpp;
class GPanel;
class Graphable;
class Point2;
class Circle;

class RUDrawComponent : public RUGraph
{
private:
	float penWidth;
	bool clickMode;
	Circle* prevCircle;

	std::vector<Circle*> circles;//when RUGraph is template type use this

protected:
	// events
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);

public:
	static const int MODE_CIRCLES = 0;
	static const int MODE_NELLIPSE = 1;

	// constructors & destructor
	RUDrawComponent(int, int, int);
	~RUDrawComponent();

	// gets
	int getMode() const;
	float getPenWidth() const;

	// sets
	void toggleMode();
	void setPenWidth(float);

	// render
	virtual void updateBackground(gfxpp*);
	virtual shmea::GString getType() const;
	void clear(bool = false);

	// Circle functions
	void addCircle(const Point2*, double);

	//
	virtual void add(gfxpp*, const shmea::GString&, const Point2&);
	virtual void set(gfxpp*, const shmea::GString&, const std::vector<Point2*>&,
			 SDL_Color = RUColors::DEFAULT_COLOR_LINE);
	virtual void set(gfxpp*, const shmea::GString&, const shmea::GList&,
			 SDL_Color = RUColors::DEFAULT_COLOR_LINE);
};

#endif
