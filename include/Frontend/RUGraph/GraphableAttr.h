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
#ifndef _GRAPHABLEATTR_H
#define _GRAPHABLEATTR_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "../GFXUtilities/point2.h"
#include "../GFXUtilities/Candle.h"
#include "../Graphics/graphics.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/standardizable.h"

class RUGraph;

class GraphableAttr : public shmea::GStandardizable
{
protected:
	RUGraph* parent;
	bool xMode;
	bool yMode;
	SDL_Color lineColor;

public:

	const static int TEXTURE_MAX_DIM = 16384;

	// constructors & destructor
	GraphableAttr();
	GraphableAttr(RUGraph*, SDL_Color);
	virtual ~GraphableAttr();

	// gets
	float getXMinModed() const;
	float getXMaxModed() const;
	float getYMinModed() const;
	float getYMaxModed() const;
	bool getXMode() const;
	bool getYMode() const;
	SDL_Color getColor() const;

	// sets
	void setXMode(bool);
	void setYMode(bool);
	void setColor(SDL_Color);
	virtual void clear();

	// render
	virtual void draw(gfxpp*) = 0;
	virtual void updateBackground(gfxpp*);
};

#endif
