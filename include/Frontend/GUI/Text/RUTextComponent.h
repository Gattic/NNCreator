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
#ifndef _RUTEXTCOMPONENT
#define _RUTEXTCOMPONENT

#include "../../GItems/RUComponent.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

class gfxpp;
class GFont;

class SlidingFocusWindow
{
public:
	unsigned int index;
	unsigned int maxLen;
	unsigned int cursorIndex; // innerIndex
	unsigned int cursorDirection;

	SlidingFocusWindow()
	{
		reset();
	}

	void reset()
	{
		index = 0;
		maxLen = 0;
		cursorIndex = 0;
		cursorDirection = 0;
	}
};

class RUTextComponent : public RUComponent
{
protected:
	static const int CUROSR_LEFT = 0;
	static const int CURSOR_RIGHT = 1;

	int FONT_COLOR;

	SlidingFocusWindow cursor;

	shmea::GString text;
	shmea::GString strDrawText;
	float strWidth;
	int cursorX;
	int xClick;
	float dimRatio;
	char passwordChar;
	bool passwordField;
	unsigned int cursorStart;
	bool readOnly;

	// render
	void calculateRenderInfo(GFont*);

	// events
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);
	virtual void onKey(gfxpp*, char);

	// event listener
	void (GPanel::*KeyListener)(char);

public:
	// constructor
	RUTextComponent();
	virtual ~RUTextComponent();

	// gets
	shmea::GString getText() const;
	char getPasswordChar() const;
	bool isPasswordField() const;
	bool getReadOnly() const;

	// sets
	void setText(const char*);
	void setText(shmea::GType);
	void setPasswordChar(char);
	void setPasswordField(bool);
	void setReadOnly(bool);
	void setFontColor(int);

	// render
	void drawText(gfxpp*);
	void drawCursor(gfxpp*, float);
	virtual void updateBackground(gfxpp*) = 0;

	// event functions
	void setKeyListener(void (GPanel::*)(char));

	// events
	virtual bool onKeyHelper(gfxpp*, GPanel*, SDL_Keycode, Uint16);

	// type
	virtual shmea::GString getType() const = 0;
};

#endif
