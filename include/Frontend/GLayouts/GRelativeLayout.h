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
#ifndef _GRELATIVELAYOUT
#define _GRELATIVELAYOUT

#include "../GItems/GLayout.h"
#include <stdio.h>
#include <stdlib.h>

class gfxpp;
class EventTracker;

/*!
 * @brief GRelativeLayout
 * @details The GRelativeLayout fits as many GItems as possible in the width of the layout before
 * moving to the next row.
 */

class GRelativeLayout : public GLayout
{
	// render
	void updateBackground(gfxpp*);

public:
	GRelativeLayout(shmea::GString);

	virtual void calculateSubItemPositions(std::pair<int, int>);

	// events
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int);

	// render
	virtual void updateBackgroundHelper(gfxpp*);

	virtual shmea::GString getType() const;
};

#endif
