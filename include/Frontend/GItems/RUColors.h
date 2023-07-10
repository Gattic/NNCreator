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
	static SDL_Color COLOR_YELLOW;
	static SDL_Color COLOR_PURPLE;
	static SDL_Color COLOR_ORANGE;
	static SDL_Color COLOR_GRAY;
	static SDL_Color COLOR_LIGHT_GRAY;
	static SDL_Color COLOR_DARK_GRAY;
	static SDL_Color COLOR_CYAN;
	static SDL_Color COLOR_PINK;
	static SDL_Color COLOR_MAGENTA;
	static SDL_Color COLOR_BROWN;
	static SDL_Color COLOR_TURQUOISE;
	static SDL_Color COLOR_GOLD;
	static SDL_Color COLOR_SILVER;
	static SDL_Color COLOR_BRONZE;
	static SDL_Color COLOR_TEAL;
	static SDL_Color COLOR_NAVY;
	static SDL_Color COLOR_MAROON;
	static SDL_Color COLOR_OLIVE;
	static SDL_Color COLOR_LIME;
	static SDL_Color COLOR_AQUA;
	static SDL_Color COLOR_FUCHSIA;
	static SDL_Color COLOR_PURPLE2;
	static SDL_Color COLOR_SILVER2;
	static SDL_Color COLOR_CHOCOLATE;
	static SDL_Color COLOR_CHARTREUSE;
	static SDL_Color COLOR_INDIGO;
	static SDL_Color COLOR_TAN;
	static SDL_Color COLOR_BEIGE;
	static SDL_Color COLOR_AZURE;
	static SDL_Color COLOR_IVORY;
	static SDL_Color COLOR_LAVENDER;
	static SDL_Color COLOR_LAVENDER_BLUSH;
	static SDL_Color COLOR_LAWN_GREEN;
	static SDL_Color COLOR_LEMON_CHIFFON;
	static SDL_Color COLOR_LIGHT_BLUE;
	static SDL_Color COLOR_LIGHT_CORAL;
	static SDL_Color COLOR_LIGHT_CYAN;
	static SDL_Color COLOR_LIGHT_GOLDENROD;
	static SDL_Color COLOR_LIGHT_PINK;
	static SDL_Color COLOR_LIGHT_SALMON;
	static SDL_Color COLOR_LIGHT_SEA_GREEN;
	static SDL_Color COLOR_LIGHT_SKY_BLUE;
	static SDL_Color COLOR_LIGHT_SLATE_GRAY;
	static SDL_Color COLOR_LIGHT_STEEL_BLUE;
	static SDL_Color COLOR_LIGHT_YELLOW;
	static SDL_Color COLOR_LIGHT_RED;
	static SDL_Color COLOR_LIGHT_GREEN;
	static SDL_Color COLOR_LIGHT_ORANGE;
	static SDL_Color COLOR_LIGHT_PURPLE;
	static SDL_Color COLOR_LIGHT_BROWN;
	static SDL_Color COLOR_LIGHT_ORCHID;
	static SDL_Color COLOR_LIGHT_CORNFLOWER_BLUE;
	static SDL_Color COLOR_LIGHT_SKYBLUE;
	static SDL_Color COLOR_LIGHT_AQUAMARINE;
	static SDL_Color COLOR_LIGHT_EGGPLANT;
	static SDL_Color COLOR_LIME_GREEN;
	static SDL_Color COLOR_LINEN;
	static SDL_Color COLOR_MEDIUM_AQUAMARINE;
	static SDL_Color COLOR_MEDIUM_BLUE;
	static SDL_Color COLOR_MEDIUM_ORCHID;
	static SDL_Color COLOR_MEDIUM_PURPLE;
	static SDL_Color COLOR_MEDIUM_SEA_GREEN;
	static SDL_Color COLOR_MEDIUM_SLATE_BLUE;
	static SDL_Color COLOR_MEDIUM_SPRING_GREEN;
	static SDL_Color COLOR_MEDIUM_TURQUOISE;
	static SDL_Color COLOR_MEDIUM_VIOLET_RED;
	static SDL_Color COLOR_MEDIUM_WOOD;
	static SDL_Color COLOR_MEDIUM_YELLOW;
	static SDL_Color COLOR_MIDNIGHT_BLUE;
	static SDL_Color COLOR_MINT_CREAM;
	static SDL_Color COLOR_MISTY_ROSE;
	static SDL_Color COLOR_MOCCASIN;
	static SDL_Color COLOR_NAVAJO_WHITE;
	static SDL_Color COLOR_OLD_LACE;
	static SDL_Color COLOR_OLIVE_DRAB;
	static SDL_Color COLOR_DARK_OLIVE_GREEN;
	static SDL_Color COLOR_DARK_ORANGE;
	static SDL_Color COLOR_DARK_ORCHID;
	static SDL_Color COLOR_DARK_RED;
	static SDL_Color COLOR_DARK_SALMON;
	static SDL_Color COLOR_DARK_SEA_GREEN;
	static SDL_Color COLOR_DARK_SLATE_BLUE;
	static SDL_Color COLOR_DARK_SLATE_GRAY;
	static SDL_Color COLOR_DARK_TURQUOISE;
	static SDL_Color COLOR_DARK_VIOLET;
	static SDL_Color COLOR_DARK_YELLOW;
	static SDL_Color COLOR_DARK_PINK;
	static SDL_Color COLOR_DARK_GREEN;
	static SDL_Color COLOR_DARK_PURPLE;
	static SDL_Color COLOR_DARK_LIME_GREEN;
	static SDL_Color COLOR_DARK_CYAN;
	static SDL_Color COLOR_DARK_BROWN;
	static SDL_Color COLOR_DARK_JUNGLE_GREEN;
	static SDL_Color COLOR_DARK_QUARTZ;
	static SDL_Color COLOR_DARK_ZOMBIE_FLESH;
	static SDL_Color COLOR_DARK_OCEAN_BLUE;
	static SDL_Color COLOR_DEEP_PINK;
	static SDL_Color COLOR_DEEP_SKY_BLUE;
	static SDL_Color COLOR_DIM_GRAY;
	static SDL_Color COLOR_DODGER_BLUE;
	static SDL_Color COLOR_FIREBRICK;
	static SDL_Color COLOR_FLORAL_WHITE;
	static SDL_Color COLOR_FOREST_GREEN;
	static SDL_Color COLOR_GAINSBORO;
	static SDL_Color COLOR_GHOST_WHITE;
	static SDL_Color COLOR_YELLOW_GREEN;
	static SDL_Color COLOR_HONEYDEW;
	static SDL_Color COLOR_HOT_PINK;
	static SDL_Color COLOR_INDIAN_RED;
	static SDL_Color COLOR_TRANSPARENT;
	static SDL_Color COLOR_NONE;
};

#endif
