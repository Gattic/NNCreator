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

#ifndef _RUCOMPONENT
#define _RUCOMPONENT

#include "GItem.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

class gfxpp;
class EventTracker;

class RUComponent : public GItem
{
protected:
	virtual void updateBackground(gfxpp*) = 0;

public:
	RUComponent();

	virtual void calculateSubItemPositions(std::pair<int, int>);

	// events
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int);
	virtual void hover(gfxpp*);
	virtual void unhover(gfxpp*);

	// render
	virtual void updateBackgroundHelper(gfxpp*);

	// type
	virtual shmea::GString getType() const = 0;
};

#endif
