
#ifndef _GFXPP3D
#define _GFXPP3D

#include "../GFXUtilities/quaternion.h"
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
#include <string>

class GItem;
class RUComponent;
class RULabel;
class GPanel;
class Object;
class GFont;

class gfxpp3D
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
	int errorFlag;
	bool running;
	int width;
	int height;
	float hunterZolomon; // zoom
	int renderStatus;

	int32_t frames;
	bool rotate;
	bool move;
	int32_t now;
	int32_t then;

	// for mouse
	int mouseX;
	int mouseY;

	// for key presses
	bool CTRLPressed;
	bool ALTPressed;
	bool spacePressed;
	bool fPressed;
	bool uPressed;
	bool qPressed;
	bool gPressed;
	bool rPressed;
	bool lPressed;
	bool upPressed;
	bool downPressed;
	bool leftPressed;
	bool rightPressed;

	Quaternion roll;
	Quaternion pitch;
	Quaternion yaw;

	SDL_Window* window;
	SDL_GLContext context;
	SDL_Renderer* renderer;

	std::vector<GItem*> guiElements; // < RUComponent* || GLayout* >

	SDL_Cursor* systemCursor;
	GItem* focusedItem;
	std::vector<Object*> objects;
	unsigned int cObjIndex;


	// main
	void display();
	int initHelper(bool, shmea::GString, bool);
	void init3D();
	void clean3D();

public:
	static const float MAX_FRAMES_PER_SECOND = 30.0f;

	static const int X_AXIS = 0;
	static const int Y_AXIS = 1;
	static const int Z_AXIS = 2;

	gfxpp3D();
	gfxpp3D(shmea::GString, bool = true, bool = true, int = 800, int = 600);
	int getErrorFlag() const;
	SDL_Renderer* getRenderer();

	// GFX Utils
	static unsigned int RGBfromHue(double, int8_t*, int8_t*, int8_t*);
	bool contains(const Object*) const;
	bool contains(const Object) const;


	void addGradient(int, int, int);
	int getWidth() const;
	int getHeight() const;

	void addBasis();
	void addCube();

	// main
	void run();
	void finish();
	bool getRunning() const;
};

#endif
