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

#ifndef _GITEM
#define _GITEM

#include "Mini/RUBackgroundComponent.h"
#include "Mini/RUBorderComponent.h"
#include "Mini/RUKeyDown.h"
#include "Mini/RUKeyUp.h"
#include "Mini/RULoseFocus.h"
#include "Mini/RUMouseDown.h"
#include "Mini/RUMouseMotion.h"
#include "Mini/RUMouseUp.h"
#include "Mini/RUMouseWheel.h"
#include "RUItemArea.h"
#include "GeneralListener.h"
#include <SDL2/SDL.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

class gfxpp;
class GPanel;
class EventTracker;

class GItem : public RUBackgroundComponent,
			  public RUBorderComponent,
			  public RUKeyDown,
			  public RUKeyUp,
			  public RUMouseDown,
			  public RUMouseUp,
			  public RUMouseMotion,
			  public RUMouseWheel,
			  public RULoseFocus
{
protected:

	int id;
	unsigned int zindex;
	shmea::GString name;
	SDL_Texture* background;
	std::vector<GItem*> subitems;

	EventTracker* eventsStatus;

	// render
	virtual void updateBackground(gfxpp*) = 0;

public:

	static const unsigned int Z_FRONT = -1; // max unsigned int
	static const unsigned int Z_BACK = -2; // max unsigned int -1

	GItem();
	GItem(int, int, int, int);
	virtual ~GItem();

	// gets
	int getID() const;
	shmea::GString getName() const;
	GItem* getItemByID(int);
	GItem* getItemByName(const shmea::GString&);
	unsigned int getZIndex() const;
	SDL_Texture* getBackground();
	std::vector<GItem*> getItems() const;

	// sets
	void setID(int);
	void setName(const shmea::GString&);
	void setWidth(int);
	void setHeight(int);
	void setZIndex(unsigned int);

	// subcomps
	virtual void addSubItem(GItem*, unsigned int = Z_FRONT);
	void removeItem(gfxpp*, int);
	void removeItem(gfxpp*, const shmea::GString&);
	void clearItems(unsigned int = 0);

	virtual void calculateSubItemPositions(std::pair<int, int>) = 0;

	// render
	virtual void updateBackgroundHelper(gfxpp*) = 0;

	// event functions
	EventTracker* processEvents(gfxpp*, GPanel*, SDL_Event, int, int);
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int) = 0;

	virtual shmea::GString getType() const = 0;
};

#endif
