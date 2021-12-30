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
#ifndef _RULINEGRAPH
#define _RULINEGRAPH

#include "RUGraph.h"
#include "../GItems/RUColors.h"
#include "../GFXUtilities/point2.h"
#include "Graphable.h"
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class gfxpp;
//class Graphable<Point2>;
class Point2;

namespace shmea {
class GList;
}; // namespace shmea

class RULineGraph : public RUGraph
{
private:
	std::map<shmea::GString, Graphable<Point2>*> lines;

protected:
	// render
	virtual void updateBackground(gfxpp*);

public:

	// constructors & destructor
	RULineGraph(int, int, int = QUADRANTS_ONE);
	virtual ~RULineGraph();

	virtual shmea::GString getType() const;
	virtual void add(shmea::GString, const Point2*, SDL_Color = RUColors::DEFAULT_COLOR_LINE, bool = true);
	virtual void set(const shmea::GString&, const std::vector<Point2*>&, SDL_Color = RUColors::DEFAULT_COLOR_LINE);

	virtual void update();
	void clear(bool = false);
};

#endif
