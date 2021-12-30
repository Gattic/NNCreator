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
#ifndef _RULISTBOX
#define _RULISTBOX

#include "../GItems/RUComponent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class gfxpp;
class RULabel;
class RUScrollbar;

class RUListbox : public RUComponent
{
	friend class RUDropdown;

protected:
	bool multiSelectEnabled;
	unsigned int optionsShown;
	std::vector<RULabel*> items;
	unsigned int itemHovered;
	std::vector<unsigned int> itemsSelected;
	RUScrollbar* scrollbar;
	bool hoverDraw;

	// events
	virtual void onMouseMotion(gfxpp*, GPanel*, int, int);
	virtual void onMouseDown(gfxpp*, GPanel*, int, int);
	virtual void onMouseWheel(gfxpp*, GPanel*, int, int, int);

public:
	static const int DEFAULT_SCROLLBAR_WIDTH = 20;

	// constructors & destructor
	RUListbox();
	~RUListbox();

	// gets
	bool getMultiSelect() const;
	unsigned int getOptionsShown() const;
	const std::vector<unsigned int> getItemsSelected() const;
	unsigned int size() const;

	// sets
	void setWidth(int);
	void setHeight(int);
	void setMultiSelect(bool);
	void setOptionsShown(unsigned int);
	void addOption(shmea::GString);
	void addSelection(unsigned int);
	void clearSelections();
	void clearOptions();
	// void clear();

	// render
	virtual void updateBackground(gfxpp*);
	void updateLabels();
	virtual shmea::GString getType() const;
	virtual void unhover();
};

#endif
