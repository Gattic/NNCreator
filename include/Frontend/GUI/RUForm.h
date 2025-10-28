// Copyright 2025 Robert Carneiro
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
#ifndef _RUFORM
#define _RUFORM

#include "../GItems/RUComponent.h"

class RUForm : public RUComponent
{
protected:
    // render
    virtual void updateBackground(gfxpp*);

    // events
    virtual void onKeyDown(gfxpp*, GPanel*, GfxKeycode, Uint16);

    // helpers
    int findFocusedIndex() const;
    bool isFocusable(GItem*) const;
    void focusByIndex(gfxpp*, int);
    void focusNext(gfxpp*);
    void focusPrev(gfxpp*);

public:
    RUForm(shmea::GString);
    virtual ~RUForm();

    virtual void calculateSubItemPositions(std::pair<int, int>);
    virtual void processSubItemEvents(gfxpp*, EventTracker*, GPanel*, GfxEvent, int, int);
    virtual void updateBackgroundHelper(gfxpp*);

    virtual shmea::GString getType() const;
};

#endif



