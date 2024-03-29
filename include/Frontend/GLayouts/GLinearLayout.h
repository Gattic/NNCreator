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
#ifndef _GLINEARLAYOUT
#define _GLINEARLAYOUT

#include "../GItems/GLayout.h"
#include <stdio.h>
#include <stdlib.h>

class gfxpp;
class EventTracker;

/*!
 * @brief GLinearLayout
 * @details The GLinearLayout fits one GItem per row.
 */

class GLinearLayout : public GLayout
{
protected:
	int orientation;

	// render
	virtual void updateBackground(gfxpp*);

public:
	const static int VERTICAL = 0;
	const static int HORIZONTAL = 1;

	GLinearLayout(shmea::GString, int = VERTICAL);

	int getOrientation() const;
	void setOrientation(int);

	virtual void calculateSubItemPositions(std::pair<int, int>);

	// events
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int);

	// render
	virtual void updateBackgroundHelper(gfxpp*);

	virtual shmea::GString getType() const;
};

#endif
