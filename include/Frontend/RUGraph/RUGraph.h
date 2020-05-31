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
#ifndef _RUGRAPH
#define _RUGRAPH

#include "../GItems/RUColors.h"
#include "../GItems/RUComponent.h"
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class gfxpp;
class Graphable;
class Point2;
class RULabel;

namespace shmea {
class GList;
class GTable;
};

class RUGraph : public RUComponent
{
private:
	int graphSize;
	int axisOriginX;
	int axisOriginY;
	int axisWidth;
	bool gridEnabled;
	int gridLineWidth;
	int quadrants;

protected:
	// std::vector<GraphLine*> lines;
	std::map<std::string, Graphable*> lines;
	pthread_mutex_t* plotMutex;
	RULabel* titleLabel;

	// render
	virtual void updateBackground(gfxpp*);

public:
	static const unsigned int DEFAULT_NUM_ZONES = 10;

	static const unsigned int DEFAULT_GRAPH_SIZE = 1;
	static const unsigned int DEFAULT_AXIS_WIDTH = 3;
	static const unsigned int DEFAULT_GRIDLINE_WIDTH = 1;

	static const int QUADRANTS_ONE = 0;
	static const int QUADRANTS_FOUR = 1;

	// constructors & destructor
	RUGraph(int, int, int = QUADRANTS_ONE);
	~RUGraph();

	// gets
	int getGraphSize() const;
	int getAxisOriginX() const;
	int getAxisOriginY() const;
	int getAxisWidth() const;
	bool getGridEnabled() const;
	int getGridLineWidth() const;
	int getQuadrants() const;
	float getQuadrantOffsetX() const;
	float getQuadrantOffsetY() const;

	// sets
	void setGraphSize(int);
	void setAxisWidth(int);
	void setGridEnabled(bool);
	void setGridLineWidth(int);
	void setQuadrants(int);
	void setTitleLabel(std::string);

	virtual std::string getType() const;
	void setPoints(const std::string&, const std::vector<Point2*>&, int = 0,
				   SDL_Color = RUColors::DEFAULT_COLOR_LINE);
	void setLine(const std::string&, const shmea::GList&, int = 0,
				 SDL_Color = RUColors::DEFAULT_COLOR_LINE);
	void addScatterPoints(const shmea::GTable&);
	void buildDotMatrix();
	void clear(bool = false);
};

#endif
