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

class RUTextComponent : public RUComponent
{
protected:
	static const int IS_LEFT = 0;
	static const int IS_RIGHT = 1;

	static std::string FONT_PATH;
	static const int DEFAULT_FONT_SIZE = 30; // font resolution?
	static TTF_Font* font;
	int fontSize;

	std::string text;
	std::string strDrawText;
	float strWidth;
	float cursorXGap;
	float cursorYGap;
	int cursorX;
	float dimRatio;
	unsigned int boxInnerIndex;
	unsigned int boxIndex;
	unsigned int boxLen;
	char passwordChar;
	bool passwordField;
	unsigned int cursorStart;
	bool readOnly;
	SDL_Color textColor;

	// render
	void calculateRenderInfo();

	// events
	virtual void onMouseDown(GPanel*, int, int);
	virtual void onKey(char);

	// event listener
	void (GPanel::*KeyListener)(char);

public:
	// constructor
	RUTextComponent();
	~RUTextComponent();

	// gets
	std::string getText() const;
	SDL_Color getTextColor() const;
	char getPasswordChar() const;
	bool isPasswordField() const;
	bool getReadOnly() const;
	int getFontSize() const;

	// sets
	void setText(const char*);
	void setText(std::string);
	void setPasswordChar(char);
	void setPasswordField(bool);
	void setReadOnly(bool);
	void setTextColor(SDL_Color);
	void setFontSize(int);

	//
	static bool validChar(char);
	static char keycodeTOchar(SDL_Keycode);
	static char specialChar(char keyPressed);

	// render
	void drawText(SDL_Renderer*);
	void drawCursor(SDL_Renderer*);
	virtual void updateBackground(SDL_Renderer*) = 0;

	// event functions
	void setKeyListener(void (GPanel::*)(char));

	// events
	virtual bool onKeyHelper(GPanel*, SDL_Keycode, Uint16);

	// type
	virtual std::string getType() const = 0;
};

#endif
