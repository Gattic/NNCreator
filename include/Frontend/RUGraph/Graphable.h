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
#include "GraphableAttr.h"

class RUGraph;

template <class T>
class Graphable : public GraphableAttr
{
private:
	std::vector<T*> points;
	bool visible;

public:

	std::vector<T*> normalizedPoints;

	const static int TEXTURE_MAX_DIM = 16384;

	// constructors & destructor
	Graphable();
	Graphable(RUGraph*, SDL_Color);
	virtual ~Graphable();

	unsigned int size() const;
	unsigned int normalizedSize() const;
	bool isVisible() const;
	void setVisible(bool);
	void add(const T*, bool = true);
	void set(const std::vector<T*>&);
	virtual void clear();

	// render
	virtual void draw(gfxpp*);
	virtual void computeAxisRanges(bool = false);
};

template <class T>
Graphable<T>::Graphable(RUGraph* newParent, SDL_Color newColor) : GraphableAttr(newParent, newColor)
{
	visible = true;
}

template <class T>
Graphable<T>::Graphable()
{
	//
}

template <class T>
Graphable<T>::~Graphable()
{
	clear();
}

template <class T>
unsigned int Graphable<T>::size() const
{
	return points.size();
}

template <class T>
unsigned int Graphable<T>::normalizedSize() const
{
	return normalizedPoints.size();
}

template <class T>
bool Graphable<T>::isVisible() const
{
	return visible;
}

template <class T>
void Graphable<T>::setVisible(bool newVisible)
{
	visible = newVisible;
}

template <class T>
void Graphable<T>::set(const std::vector<T*>& newPoints)
{
	if (newPoints.empty())
		return;

	points = newPoints;
	computeAxisRanges();//sets redoRange automatically
}

// This function is recommended for TimeSeries optimizations.
template <class T>
void Graphable<T>::add(const T* newPoint, bool recompute)
{
	if (!newPoint)
		return;

	points.push_back(new T(*newPoint));
	if(recompute)
		computeAxisRanges(true);
}

/*
 * @brief draw point
 * @details draws a filled point on the renderer at (x,y) with radius r using the Midpoint circle
 * algorithm. Scatter Helper fnc
 * @param cGfx the central graphics object
 * @param cx the desired central x-coordinate
 * @param cy the desired central y-coordinate
 * @param r the desired circle radius
 */
/*template <class T>
void Graphable<T>::drawPoint(gfxpp* cGfx, int cx, int cy, int r)//Turn these 3 points into a class?
{
	if (r == 0)
		r = pointSize / 2;
	for (int i = 1; i <= r; ++i)
		drawPointOutline(cGfx, cx, cy, i);
}*/

template <class T>
void Graphable<T>::clear()
{
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		T* cPoint = points[i];
		if (!cPoint)
			continue;
		delete cPoint;
		points.erase(points.begin() + i);
	}

	points.clear();
	normalizedPoints.clear();
	visible = false;
}

#endif
