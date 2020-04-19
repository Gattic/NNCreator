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
#ifndef _GRAPHABLE_H
#define _GRAPHABLE_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class RUGraph;
class Point2;

namespace shmea {
class GList;
};

class Graphable
{
protected:
	std::vector<Point2*> points;
	float x_max, x_min, y_max, y_min;
	RUGraph* parent;

private:
	SDL_Color lineColor;
	pthread_mutex_t* plotMutex;

	void computeAxisRanges();

public:
	static const int LINE = 0;
	static const int SCATTER = 1;

	// constructors & destructor
	Graphable(RUGraph*, SDL_Color);
	virtual ~Graphable();

	// gets
	SDL_Color getColor() const;

	// sets
	void setParent(RUGraph*);
	void setColor(SDL_Color);
	void setPoints(const std::vector<Point2*>&);
	void setLine(const shmea::GList&);
	virtual void clear();

	// render
	virtual void updateBackground(SDL_Renderer*);
	virtual void draw(SDL_Renderer*) = 0;
	virtual std::string getType() const = 0;
};

#endif
