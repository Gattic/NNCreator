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
#ifndef _GPANEL
#define _GPANEL

#include "GItem.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Database/GString.h"
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>
#include <queue>

class gfxpp;
class RUComponent;
class GLayout;

// GPanel is highest-level abstraction for rendering.
class GPanel : public GItem
{
protected:

	std::queue<const shmea::ServiceData*> updateQueue;
	pthread_mutex_t* qMutex;

	void processQ(gfxpp*);
	void popQ();
	virtual void updateFromQ(const shmea::ServiceData*);

	// Lifetime (virtual) functions
	virtual void onStart() = 0;
	virtual void onShow(gfxpp*);
	virtual void onHide(gfxpp*);

	// render
	virtual void updateBackground(gfxpp*);

public:

	GPanel(const shmea::GString&, int, int);
	virtual ~GPanel();

	void addToQ(const shmea::ServiceData*);
	virtual void addSubItem(GItem*, unsigned int = Z_FRONT);
	virtual void calculateSubItemPositions(std::pair<int, int>);

	// events
	void show(gfxpp*);
	void hide(gfxpp*);
	virtual void hover(gfxpp*);
	virtual void unhover(gfxpp*);
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int);
	virtual void updateBackgroundHelper(gfxpp*);

	virtual shmea::GString getType() const;

	void MsgBox(shmea::GString, shmea::GString, int, GeneralListener = GeneralListener());
};

#endif
