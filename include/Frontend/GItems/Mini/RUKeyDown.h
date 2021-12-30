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
#ifndef _RUKEYDOWN
#define _RUKEYDOWN

#include "../RUItemArea.h"
#include "../GeneralListener.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

typedef void (*KeyDownEvent)(const shmea::GString&);

class gfxpp;
class GItem;
class GPanel;
class EventTracker;

class RUKeyDown : public virtual RUItemArea
{
protected:
	// events
	virtual void onKeyDown(gfxpp*, GPanel*, SDL_Keycode, Uint16);

	// event listeners
	GeneralListener KeyDownListener;

public:
	// constructors & destructor
	RUKeyDown();
	virtual ~RUKeyDown();

	// event functions
	void setKeyDownListener(GeneralListener);

	// events
	void onKeyDownHelper(gfxpp*, EventTracker*, GPanel*, SDL_Keycode, Uint16);
};

#endif
