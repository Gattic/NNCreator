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

#ifndef _RUITEMAREA
#define _RUITEMAREA

#include <SDL2/SDL.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

class RUItemArea
{

protected:
	int x;
	int y;
	int width;
	int height;
	int paddingX;
	int paddingY;
	int marginX;
	int marginY;
	bool visible;
	bool drawUpdate;

	// events
	bool inRange(int, int) const;

public:
	RUItemArea();
	RUItemArea(int, int, int, int);
	virtual ~RUItemArea();

	// gets
	int getX() const;
	int getY() const;
	virtual int getWidth() const;
	virtual int getHeight() const;
	int getPaddingX() const;
	int getPaddingY() const;
	int getMarginX() const;
	int getMarginY() const;
	SDL_Rect getLocationRect() const;
	bool isVisible() const;
	bool getDrawUpdateRequired() const;

	// sets
	void setX(int);
	void setY(int);
	void setWidth(int);
	void setHeight(int);
	void setPadding(int);
	void setPaddingX(int);
	void setPaddingY(int);
	void setMargin(int);
	void setMarginX(int);
	void setMarginY(int);
	void setVisible(bool);
	void show();
	void hide();
	void requireDrawUpdate();
};

#endif
