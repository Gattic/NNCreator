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

#ifndef _GLAYOUT
#define _GLAYOUT

#include "GItem.h"
#include "Backend/Database/GString.h"
#include <SDL2/SDL.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

class gfxpp;

class GLayout : public GItem
{
protected:
	GPanel* panel;
	int layoutType; // 0 = Relative; 1 = Linear

public:
	GLayout();
	virtual ~GLayout();

	// gets
	int getLayoutType() const;

	virtual void hover(gfxpp*);
	virtual void unhover(gfxpp*);
	virtual shmea::GString getType() const = 0;
};

#endif
