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
#ifndef _RUSCROLLBAR
#define _RUSCROLLBAR

#include "../GItems/RUComponent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

class gfxpp;
class RUImageComponent;

class RUScrollbar : public RUComponent
{
protected:
	int value;
	int maxValue;
	int optionsShown;
	int orientation;
	SDL_Color barColor;
	float arrowSpace;

	const static shmea::GString arrowUpLocation;
	const static shmea::GString arrowDownLocation;
	const static shmea::GString scrollBarLocation;

	// scrollbar images
	RUImageComponent* arrowUp;
	RUImageComponent* arrowDown;
	RUImageComponent* scrollBar;

	// events
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);
	virtual void onMouseWheel(gfxpp*, GPanel*, int, int, int);

public:
	static const int ORIENTATION_HORIZONTAL = 0;
	static const int ORIENTATION_VERTICAL = 1;

	// constructors & destructor
	RUScrollbar();
	RUScrollbar(int);
	~RUScrollbar();

	// gets
	int getValue() const;
	int getMaxValue() const;
	int getOptionsShown() const;
	int getOrientation() const;

	// sets
	void setWidth(int);
	void setHeight(int);
	int increment();
	int decrement();
	void setValue(int);
	void setMaxValue(int);
	void setOptionsShown(int);
	void clear();
	void setOrientation(int);
	void calculateBarPos();

	// render
	virtual void updateBackground(gfxpp*);
	virtual shmea::GString getType() const;
};

#endif
