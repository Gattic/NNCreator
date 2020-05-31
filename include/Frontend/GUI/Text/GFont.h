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
#ifndef _GFONT
#define _GFONT

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_ttf.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>

class SDL_Renderer;
class SDL_Texture;

class GLetter
{
private:
	char letter; // unnecassary
	SDL_Texture* tex;
	int width; // NOT SCALED BY DIMRATIO YET!!
public:
	GLetter()
	{
	}

	GLetter(char newLetter, SDL_Texture* newTex, int newWidth)
	{
		letter = newLetter;
		tex = newTex;
		width = newWidth;
	}

	~GLetter()
	{
		// TODO: Destroy the texture here
	}

	char getLetter() const
	{
		return letter;
	}

	SDL_Texture* getTexture()
	{
		return tex;
	}

	int getWidth() const
	{
		return width;
	}
};

class GFont
{
private:
	static const int DEFAULT_FONT_SIZE = 30; // font resolution?

	std::string fontPath;
	TTF_Font* font;
	int fontSize;
	SDL_Color textColor;
	std::map<char, GLetter*> textureMap;
	SDL_Renderer* cRenderer;
	int maxHeight;

	void loadLetters();

public:
	// constructor
	GFont();
	GFont(SDL_Renderer*, std::string = "");
	~GFont();

	SDL_Color getTextColor() const;
	int getFontSize() const;
	std::string getFontPath() const;
	TTF_Font* getFont() const;
	GLetter* getLetter(char) const;
	int getMaxHeight() const;

	void setFontSize(int);
	void setTextColor(SDL_Color);

	//
	static bool validChar(char);
	static char keycodeTOchar(SDL_Keycode);
	static char specialChar(char keyPressed);
};

#endif
