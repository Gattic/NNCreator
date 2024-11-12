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
#ifndef _RUBACKGROUNDCOMP
#define _RUBACKGROUNDCOMP

#include "../RUItemArea.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GPointer.h"
#include "Backend/Database/image.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define DEFAULT_IMAGE_BG "resources/gui/Components/Background.bmp"
#define DEFAULT_IMAGE_BG_HIGHLIGHTED "resources/gui/Components/BackgroundHighlighted.bmp"

class gfxpp;

class RUBackgroundComponent : public virtual RUItemArea
{
protected:
	bool bgEnabled;
	bool bgColorEnabled;
	SDL_Surface* surfaceTheUSA;
	shmea::GPointer<shmea::Image> bgImage;
	//shmea::GPointer<shmea::Image> bgCache;
	SDL_Color bgColor;
	shmea::GString bgImageLocation;
	int bgImageType;

	bool resetSurface();
	void refreshImage();

	void fromSurface(SDL_Surface*);
	void fromImage(shmea::GPointer<shmea::Image>);

public:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	static const unsigned int rmask = 0xFF000000;
	static const unsigned int gmask = 0x00FF0000;
	static const unsigned int bmask = 0x0000FF00;
	static const unsigned int amask = 0x000000FF;

	static const int ALPHA_INDEX = 0;
	static const int BLUE_INDEX = 1;
	static const int GREEN_INDEX = 2;
	static const int RED_INDEX = 3;

#else
	static const unsigned int rmask = 0x000000FF;
	static const unsigned int gmask = 0x0000FF00;
	static const unsigned int bmask = 0x00FF0000;
	static const unsigned int amask = 0xFF000000;

	static const int RED_INDEX = 0;
	static const int GREEN_INDEX = 1;
	static const int BLUE_INDEX = 2;
	static const int ALPHA_INDEX = 3;
#endif

	static const int TYPE_NONE = 0;
	static const int TYPE_FILE = 1;
	static const int TYPE_GIMAGE = 2;

	// constructors & destructor
	RUBackgroundComponent();
	RUBackgroundComponent(SDL_Color);
	virtual ~RUBackgroundComponent();

	// gets
	bool getBGEnabled() const;
	bool getBGColorEnabled() const;
	shmea::GString getBGImageLocation() const;
	SDL_Color getBGColor() const;

	// sets
	void toggleBG(bool);
	void toggleBGColor(bool);
	void setBGImageFromLocation(const shmea::GString&);
	void setBGImageFromSurface(SDL_Surface*);
	void setBGImage(shmea::GPointer<shmea::Image>);
	void setBGColor(SDL_Color);

	// render
	void updateBGBackground(gfxpp*);
	void drawVerticalGradient(SDL_Renderer*, SDL_Rect, SDL_Color, SDL_Color, int = 0);
};

#endif
