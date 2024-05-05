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
#include "GeneralGraphable.h"
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class gfxpp;
class RULabel;

namespace shmea {
class GList;
class GTable;
}; // namespace shmea

class RUGraph : public RUComponent, public shmea::GStandardizable
{
private:
	int graphSize;
	int axisOriginX;
	int axisOriginY;
	int axisWidth;
	bool dotMatrixEnabled;
	bool gridEnabled;
	int gridLineWidth;
	int quadrants;

	float vscale;
	unsigned int period;
	unsigned int sourceAgg; //aggregate level of the source data
	unsigned int agg; //aggregatelevel of our data

protected:
	std::map<shmea::GString, GeneralGraphable*> graphables;

	// events
	virtual void onMouseUp(gfxpp*, GPanel*, int, int);
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);

	// render
	static void* computeGraph(void* y);
	virtual void updateBackground(gfxpp*);

public:

	const static unsigned int P_ALL = 0;
	const static unsigned int P_1D = 1;
	const static unsigned int P_5D = 2;
	const static unsigned int P_10D = 3;
	const static unsigned int P_1M = 4;
	const static unsigned int P_3M = 5;
	const static unsigned int P_6M = 6;
	const static unsigned int P_1Y = 7;
	const static unsigned int P_5Y = 8;
	const static unsigned int P_10Y = 9;

	const static unsigned int AGG_1m = 1;
	const static unsigned int AGG_2m = 2;
	const static unsigned int AGG_3m = 3;
	const static unsigned int AGG_5m = 5;
	const static unsigned int AGG_15m = 15;
	const static unsigned int AGG_30m = 30;
	const static unsigned int AGG_1h = 60;
	const static unsigned int AGG_4h = 240;
	const static unsigned int AGG_1D = 390;
	const static unsigned int AGG_1W = 1950;
	const static unsigned int AGG_3W = 5850;

	static const unsigned int DEFAULT_NUM_ZONES = 10;
	static const unsigned int DEFAULT_GRAPH_SIZE = 1;
	static const unsigned int DEFAULT_AXIS_WIDTH = 3;
	static const unsigned int DEFAULT_GRIDLINE_WIDTH = 1;

	static const int QUADRANTS_ONE = 0;
	static const int QUADRANTS_FOUR = 1;

	// constructors & destructor
	RUGraph(int, int, int = QUADRANTS_ONE);
	virtual ~RUGraph();

	template< typename T>
	void add(shmea::GString, const T*, SDL_Color = RUColors::DEFAULT_COLOR_LINE, bool = true);

	template< typename T>
	void set(const shmea::GString&, const std::vector<T*>&, SDL_Color = RUColors::DEFAULT_COLOR_LINE);


	template< typename T>
	void set(shmea::GString, const T*, SDL_Color = RUColors::DEFAULT_COLOR_LINE, bool = true);

	std::vector<shmea::GString> getNames() const;
	void remove(const shmea::GString&);

	// gets
	int getGraphSize() const;
	int getAxisOriginX() const;
	int getAxisOriginY() const;
	int getAxisWidth() const;
	bool getDotMatrixEnabled() const;
	bool getGridEnabled() const;
	int getGridLineWidth() const;
	int getQuadrants() const;
	float getQuadrantOffsetX() const;
	float getQuadrantOffsetY() const;
	float getVScale() const;
	unsigned int getPeriod() const;
	unsigned int getSourceAggregate() const;
	unsigned int getAggregate() const;
	std::map<shmea::GString, GeneralGraphable*>  getGraphables();

	// sets
	void setGraphSize(int);
	void setAxisWidth(int);
	void setDotMatrixEnabled(bool);
	void setGridEnabled(bool);
	void setGridLineWidth(int);
	void setQuadrants(int);
	void setTitleLabel(shmea::GString);
	void setVScale(float);
	void setPeriod(unsigned int);
	void setSourceAggregate(unsigned int);
	void setAggregate(unsigned int);

	virtual shmea::GString getType() const;
	virtual void update();
	virtual void clear(bool = false);
};

template< typename T>
void RUGraph::add(shmea::GString label, const T* newPoint, SDL_Color lineColor, bool recompute)
{
	T* plotterPoint = new T(*newPoint);
	if ((graphables.find(label) == graphables.end()) || (graphables[label] == NULL))
	{
		std::vector<T*> newPointVec;
		newPointVec.push_back(plotterPoint);
		set(label, newPointVec, lineColor);
		return;
	}

	GeneralGraphable* cPlotter = graphables[label];
	if(!cPlotter)
		return;

	cPlotter->add(plotterPoint, recompute);
	cPlotter->setColor(lineColor);

	// DON'T trigger the draw update here
}

template< typename T>
void RUGraph::set(const shmea::GString& label, const std::vector<T*>& graphPoints, SDL_Color lineColor)
{
	GeneralGraphable* newPlotter = NULL;
	if (graphables.find(label) != graphables.end())
	{
		newPlotter = graphables[label];
		if(!newPlotter)
			return;

		newPlotter->setColor(lineColor);
	}
	else
	{
		newPlotter = new GeneralGraphable(this, lineColor, T());
		if (!newPlotter)
			return;
		graphables[label] = newPlotter;
		if (!graphables[label])
			return;
	}

	newPlotter->set(graphPoints);

	// trigger the draw update
	drawUpdate = true;
}

template< typename T>
void RUGraph::set(shmea::GString label, const T* newPoint, SDL_Color lineColor, bool recompute)
{
    clear();
    add(label, newPoint, lineColor, recompute);
}


#endif
