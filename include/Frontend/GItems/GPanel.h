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
#include <SDL2/SDL.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

class gfxpp;
class RUComponent;
class GLayout;

// GPanel is highest-level abstraction for rendering.
class GPanel : public GItem
{

protected:
	// Lifetime (virtual) functions
	virtual void onStart() = 0;
	virtual void onShow(gfxpp*);
	virtual void onHide(gfxpp*);

	// render
	virtual void updateBackground(gfxpp*);

public:
	GPanel(const std::string&, int, int);

	virtual void addSubItem(GItem*, int = Z_FRONT);
	virtual void calculateSubItemPositions(std::pair<int, int>);

	// events
	void show(gfxpp*);
	void hide(gfxpp*);
	virtual void hover(gfxpp*);
	virtual void unhover(gfxpp*);
	virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, SDL_Event, int, int);
	virtual void updateBackgroundHelper(gfxpp*);

	virtual void GuiCommander1(const std::string&, int, int) = 0;
	virtual void GuiCommander2(const std::string&, int, int) = 0;
	virtual void GuiCommander3(const std::string&, int, int) = 0;
	virtual void GuiCommander4(const std::string&, int, int) = 0;
	virtual void GuiCommander5(const std::string&, int, int) = 0;
	virtual void GuiCommander6(const std::string&, int, int) = 0;
	virtual void GuiCommander7(const std::string&, int, int) = 0;
	virtual void GuiCommander8(const std::string&, int, int) = 0;
	virtual void GuiCommander9() = 0;
	virtual void GuiCommander10(const std::string&, int, int) = 0;
	virtual void GuiCommander11(char) = 0;
	virtual void GuiCommander12(const std::string&, int, int) = 0;
	virtual void GuiCommander13(const std::string&, int, int) = 0;
	virtual void GuiCommander14(const std::string&, int, int) = 0;
	virtual void GuiCommander15(int) = 0;

	virtual std::string getType() const;

	void MsgBox(std::string, std::string, int);
};

#endif
