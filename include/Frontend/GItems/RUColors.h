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
#ifndef _RUCOLORS
#define _RUCOLORS

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>

class RUColors
{
public:
	// General GUI
	static SDL_Color DEFAULT_COLOR_BACKGROUND;
	static SDL_Color DEFAULT_COMPONENT_BACKGROUND;
	static SDL_Color DEFAULT_COLOR_TEXT;
	static SDL_Color DEFAULT_COLOR_BAR;
	static SDL_Color DEFAULT_COLOR_SCROLLBAR;
	static SDL_Color DEFAULT_COLOR_BORDER;
	static SDL_Color DEFAULT_COLOR_CANDLE;
	static SDL_Color DEFAULT_COLOR_PROGRESS_BAR;
	static SDL_Color DEFAULT_COLOR_POINT;
	static SDL_Color DEFAULT_COMPONENT_HIGHLIGHT;
	static SDL_Color DEFAULT_DROPDOWN_ARROW;
	static SDL_Color DEFAULT_TEXT_COLOR;

	// Alternative Text Colors
	static SDL_Color TEXT_COLOR_GREEN;
	static SDL_Color TEXT_COLOR_RED;

	// Buttons
	static SDL_Color DEFAULT_BUTTON_GREEN;
	static SDL_Color DEFAULT_BUTTON_BORDER_GREEN;
	static SDL_Color DEFAULT_BUTTON_HOVER_GREEN;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_GREEN;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_GREEN;

	static SDL_Color DEFAULT_BUTTON_BLUE;
	static SDL_Color DEFAULT_BUTTON_BORDER_BLUE;
	static SDL_Color DEFAULT_BUTTON_HOVER_BLUE;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_BLUE;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_BLUE;

	static SDL_Color DEFAULT_BUTTON_RED;
	static SDL_Color DEFAULT_BUTTON_BORDER_RED;
	static SDL_Color DEFAULT_BUTTON_HOVER_RED;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_RED;
	static SDL_Color DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_RED;

	// Graph Lines
	static SDL_Color DEFAULT_COLOR_LINE;
	static SDL_Color GRAPH_LINE_2;
	static SDL_Color GRAPH_LINE_3;
	static SDL_Color GRAPH_LINE_4;
	static SDL_Color GRAPH_LINE_5;
	static SDL_Color GRAPH_LINE_6;

	// Specific Colors
	static SDL_Color COLOR_RED;
	static SDL_Color COLOR_BLUE;
	static SDL_Color COLOR_GREEN;
	static SDL_Color COLOR_WHITE;
	static SDL_Color COLOR_BLACK;
};

#endif
