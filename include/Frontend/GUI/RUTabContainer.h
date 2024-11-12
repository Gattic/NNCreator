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
#ifndef _RUTABCONTAINER
#define _RUTABCONTAINER

#include "../GItems/RUComponent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

typedef void (*OptionChangedEvent)(int);

class gfxpp;
class RUButton;

class RUTabContainer : public RUComponent
{

protected:
	int orientation;
	int tabHeight;
	bool tabsVisible;
	static const int HORIZONTAL_TABS = 0;
	static const int VERTICAL_TABS = 1;

	unsigned int optionsShown;
	unsigned int itemHovered;
	unsigned int tabSelected;
	unsigned int prevTabSelected;

	typedef std::pair<RUButton*, GItem*> RUTab;
	std::vector<RUTab> items;

	bool hoverDraw;

	// events
	void (GPanel::*OptionChangedListener)(int);
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);
	virtual void onMouseMotion(gfxpp*, GPanel*, int, int);

public:
	static const int DEFAULT_SIDE_WIDTH = 24;

	// constructors & destructor
	RUTabContainer();
	virtual ~RUTabContainer();

	// gets
	virtual int getHeight() const;
	bool getTabsVisible() const;
	unsigned int getOptionsShown() const;
	unsigned int getTabSelected();
	unsigned int size() const;

	// sets
	void setWidth(int);
	void setTabHeight(int);
	void setTabsVisible(bool);
	void setOptionsShown(unsigned int);
	void addTab(shmea::GString, GItem* = NULL);
	void clearOptions();
	void setSelectedTab(unsigned int); // int = index
	void setSelectedTab(shmea::GString);  // string = tab name

	// events
	void setOptionChangedListener(void (GPanel::*)(int));

	// render
	virtual void updateBackground(gfxpp*);
	void updateLabels();
	virtual void unhover();
	virtual shmea::GString getType() const;
};

#endif
