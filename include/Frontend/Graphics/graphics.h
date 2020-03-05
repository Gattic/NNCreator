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
#ifndef _GRAPHICS
#define _GRAPHICS

#include "../GItems/GItem.h"
#include "../GItems/GLayout.h"
#include "../GItems/GPanel.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class GItem;
class RUComponent;
class RULabel;
class GPanel;
class Object;
class Quaternion;

class Graphics
{
	friend class RUComponent;
	friend class RUTextComponent;
	friend class RUImageComponent;
	friend class RUButton;
	friend class RULabel;
	friend class RUTextbox;
	friend class RUGraph;
	friend class RUProgressBar;
	friend class RUScrollbar;
	friend class RUListbox;
	friend class RUDropdown;
	friend class RUKeyDown;
	friend class RUKeyUp;

private:
	static bool running;
	static int width;
	static int height;
	static float hunterZolomon; // zoom
	static int renderStatus;

	static int32_t frames;
	static bool rotate;
	static bool move;
	static int32_t now;
	static int32_t then;

	// for mouse
	static int mouseX;
	static int mouseY;

	// for key presses
	static bool CTRLPressed;
	static bool ALTPressed;
	static bool spacePressed;
	static bool fPressed;
	static bool uPressed;
	static bool qPressed;
	static bool gPressed;
	static bool rPressed;
	static bool lPressed;
	static bool upPressed;
	static bool downPressed;
	static bool leftPressed;
	static bool rightPressed;

	static Quaternion roll;
	static Quaternion pitch;
	static Quaternion yaw;

	static SDL_Window* window;
	static SDL_GLContext context;
	static SDL_Renderer* renderer;

	static std::vector<GItem*> guiElements; // < RUComponent* || GLayout* >

	static GItem* focusedItem;
	static std::vector<Object*> objects;
	static unsigned int cObjIndex;

	// default gui
	static RULabel* fpsLabel;

	// main
	static void display();
	static int initHelper(bool);
	static int init2D();
	static void init3D();
	static void clean2D();
	static void clean3D();

public:
	static const float MAX_FRAMES_PER_SECOND = 30.0f;

	static const int _2D = 0;
	static const int _3D = 1;

	static const int X_AXIS = 0;
	static const int Y_AXIS = 1;
	static const int Z_AXIS = 2;

	// GFX Utils
	static unsigned int RGBfromHue(double, int8_t*, int8_t*, int8_t*);
	static bool contains(const Object*);
	static bool contains(const Object);

	// 2D
	static GPanel* focusedPanel;
	static void addGradient(int, int, int);
	static void addItem(GItem*);
	static void removeItem(int); // id
	static GItem* getItemByID(int);
	static void setFocus(GItem*);
	static int getWidth();
	static int getHeight();
	static void MsgBox(std::string, std::string, int);

	// 3D
	static void addBasis();
	static void addCube();

	// main
	static int init(int, int, int);
	static int init(int);
	static void run();
	static void changeRenderStatus(int);
	static void finish();
};

#endif
