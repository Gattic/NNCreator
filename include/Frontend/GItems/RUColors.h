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

#include "../Graphics/GfxTypes.h"
#include <stdio.h>
#include <stdlib.h>

class RUColors
{
public:
	// General GUI
	static GfxColor DEFAULT_COLOR_BACKGROUND;
	static GfxColor DEFAULT_COMPONENT_BACKGROUND;
	static GfxColor DEFAULT_COLOR_TEXT;
	static GfxColor DEFAULT_COLOR_BAR;
	static GfxColor DEFAULT_COLOR_SCROLLBAR;
	static GfxColor DEFAULT_COLOR_BORDER;
	static GfxColor DEFAULT_COLOR_CANDLE;
	static GfxColor DEFAULT_COLOR_PROGRESS_BAR;
	static GfxColor DEFAULT_COLOR_POINT;
	static GfxColor DEFAULT_COMPONENT_HIGHLIGHT;
	static GfxColor DEFAULT_DROPDOWN_ARROW;
	static GfxColor DEFAULT_TEXT_COLOR;

	// Alternative Text Colors
	static GfxColor TEXT_COLOR_GREEN;
	static GfxColor TEXT_COLOR_RED;

	// Buttons
	static GfxColor DEFAULT_BUTTON_GREEN;
	static GfxColor DEFAULT_BUTTON_BORDER_GREEN;
	static GfxColor DEFAULT_BUTTON_HOVER_GREEN;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_GREEN;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_GREEN;

	static GfxColor DEFAULT_BUTTON_BLUE;
	static GfxColor DEFAULT_BUTTON_BORDER_BLUE;
	static GfxColor DEFAULT_BUTTON_HOVER_BLUE;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_BLUE;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_BLUE;

	static GfxColor DEFAULT_BUTTON_RED;
	static GfxColor DEFAULT_BUTTON_BORDER_RED;
	static GfxColor DEFAULT_BUTTON_HOVER_RED;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_RED;
	static GfxColor DEFAULT_BUTTON_COLOR_TEXT_HIGHLIGHT_RED;

	// Graph Lines
	static GfxColor DEFAULT_COLOR_LINE;
	static GfxColor GRAPH_LINE_2;
	static GfxColor GRAPH_LINE_3;
	static GfxColor GRAPH_LINE_4;
	static GfxColor GRAPH_LINE_5;
	static GfxColor GRAPH_LINE_6;

	// Specific Colors
	static GfxColor COLOR_RED;
	static GfxColor COLOR_BLUE;
	static GfxColor COLOR_GREEN;
	static GfxColor COLOR_WHITE;
	static GfxColor COLOR_BLACK;
	static GfxColor COLOR_YELLOW;
	static GfxColor COLOR_PURPLE;
	static GfxColor COLOR_ORANGE;
	static GfxColor COLOR_GRAY;
	static GfxColor COLOR_LIGHT_GRAY;
	static GfxColor COLOR_DARK_GRAY;
	static GfxColor COLOR_CYAN;
	static GfxColor COLOR_PINK;
	static GfxColor COLOR_MAGENTA;
	static GfxColor COLOR_BROWN;
	static GfxColor COLOR_TURQUOISE;
	static GfxColor COLOR_GOLD;
	static GfxColor COLOR_SILVER;
	static GfxColor COLOR_BRONZE;
	static GfxColor COLOR_TEAL;
	static GfxColor COLOR_NAVY;
	static GfxColor COLOR_MAROON;
	static GfxColor COLOR_OLIVE;
	static GfxColor COLOR_LIME;
	static GfxColor COLOR_AQUA;
	static GfxColor COLOR_FUCHSIA;
	static GfxColor COLOR_PURPLE2;
	static GfxColor COLOR_SILVER2;
	static GfxColor COLOR_CHOCOLATE;
	static GfxColor COLOR_CHARTREUSE;
	static GfxColor COLOR_INDIGO;
	static GfxColor COLOR_TAN;
	static GfxColor COLOR_BEIGE;
	static GfxColor COLOR_AZURE;
	static GfxColor COLOR_IVORY;
	static GfxColor COLOR_LAVENDER;
	static GfxColor COLOR_LAVENDER_BLUSH;
	static GfxColor COLOR_LAWN_GREEN;
	static GfxColor COLOR_LEMON_CHIFFON;
	static GfxColor COLOR_LIGHT_BLUE;
	static GfxColor COLOR_LIGHT_CORAL;
	static GfxColor COLOR_LIGHT_CYAN;
	static GfxColor COLOR_LIGHT_GOLDENROD;
	static GfxColor COLOR_LIGHT_PINK;
	static GfxColor COLOR_LIGHT_SALMON;
	static GfxColor COLOR_LIGHT_SEA_GREEN;
	static GfxColor COLOR_LIGHT_SKY_BLUE;
	static GfxColor COLOR_LIGHT_SLATE_GRAY;
	static GfxColor COLOR_LIGHT_STEEL_BLUE;
	static GfxColor COLOR_LIGHT_YELLOW;
	static GfxColor COLOR_LIGHT_RED;
	static GfxColor COLOR_LIGHT_GREEN;
	static GfxColor COLOR_LIGHT_ORANGE;
	static GfxColor COLOR_LIGHT_PURPLE;
	static GfxColor COLOR_LIGHT_BROWN;
	static GfxColor COLOR_LIGHT_ORCHID;
	static GfxColor COLOR_LIGHT_CORNFLOWER_BLUE;
	static GfxColor COLOR_LIGHT_SKYBLUE;
	static GfxColor COLOR_LIGHT_AQUAMARINE;
	static GfxColor COLOR_LIGHT_EGGPLANT;
	static GfxColor COLOR_LIME_GREEN;
	static GfxColor COLOR_LINEN;
	static GfxColor COLOR_MEDIUM_AQUAMARINE;
	static GfxColor COLOR_MEDIUM_BLUE;
	static GfxColor COLOR_MEDIUM_ORCHID;
	static GfxColor COLOR_MEDIUM_PURPLE;
	static GfxColor COLOR_MEDIUM_SEA_GREEN;
	static GfxColor COLOR_MEDIUM_SLATE_BLUE;
	static GfxColor COLOR_MEDIUM_SPRING_GREEN;
	static GfxColor COLOR_MEDIUM_TURQUOISE;
	static GfxColor COLOR_MEDIUM_VIOLET_RED;
	static GfxColor COLOR_MEDIUM_WOOD;
	static GfxColor COLOR_MEDIUM_YELLOW;
	static GfxColor COLOR_MIDNIGHT_BLUE;
	static GfxColor COLOR_MINT_CREAM;
	static GfxColor COLOR_MISTY_ROSE;
	static GfxColor COLOR_MOCCASIN;
	static GfxColor COLOR_NAVAJO_WHITE;
	static GfxColor COLOR_OLD_LACE;
	static GfxColor COLOR_OLIVE_DRAB;
	static GfxColor COLOR_DARK_OLIVE_GREEN;
	static GfxColor COLOR_DARK_ORANGE;
	static GfxColor COLOR_DARK_ORCHID;
	static GfxColor COLOR_DARK_RED;
	static GfxColor COLOR_DARK_SALMON;
	static GfxColor COLOR_DARK_SEA_GREEN;
	static GfxColor COLOR_DARK_SLATE_BLUE;
	static GfxColor COLOR_DARK_SLATE_GRAY;
	static GfxColor COLOR_DARK_TURQUOISE;
	static GfxColor COLOR_DARK_VIOLET;
	static GfxColor COLOR_DARK_YELLOW;
	static GfxColor COLOR_DARK_PINK;
	static GfxColor COLOR_DARK_GREEN;
	static GfxColor COLOR_DARK_PURPLE;
	static GfxColor COLOR_DARK_LIME_GREEN;
	static GfxColor COLOR_DARK_CYAN;
	static GfxColor COLOR_DARK_BROWN;
	static GfxColor COLOR_DARK_JUNGLE_GREEN;
	static GfxColor COLOR_DARK_QUARTZ;
	static GfxColor COLOR_DARK_ZOMBIE_FLESH;
	static GfxColor COLOR_DARK_OCEAN_BLUE;
	static GfxColor COLOR_DEEP_PINK;
	static GfxColor COLOR_DEEP_SKY_BLUE;
	static GfxColor COLOR_DIM_GRAY;
	static GfxColor COLOR_DODGER_BLUE;
	static GfxColor COLOR_FIREBRICK;
	static GfxColor COLOR_FLORAL_WHITE;
	static GfxColor COLOR_FOREST_GREEN;
	static GfxColor COLOR_GAINSBORO;
	static GfxColor COLOR_GHOST_WHITE;
	static GfxColor COLOR_YELLOW_GREEN;
	static GfxColor COLOR_HONEYDEW;
	static GfxColor COLOR_HOT_PINK;
	static GfxColor COLOR_INDIAN_RED;
	static GfxColor COLOR_TRANSPARENT;
	static GfxColor COLOR_NONE;
};

#endif
