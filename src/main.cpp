// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and
// associated documentation files (the "Software"), to deal in the Software
// without restriction,
// including without limitation the rights to use, copy, modify, merge, publish,
// distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom
// the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "main.h"
#include "Backend/Networking/main.h"
#include "Frontend/Graphics/graphics.h"
#include "crt0.h"
#include "nncreator.h"

/*!
 *@brief run program
 *@detail start up GUI for NNCreator
 *@param fullscreenMode fullscreen flag
 */
void Frontend::run(GNet::GServer* serverInstance, bool fullscreenMode, bool compatMode)
{
	int newWidth = 1200;
	int newHeight = 800;

	// Setup the gfx env
	gfxpp cGfx("NNCreator", gfxpp::_2D, fullscreenMode, compatMode, newWidth, newHeight);

	// gfx err check
	int gfxInitialized = cGfx.getErrorFlag();
	if (gfxInitialized < 0)
	{
		printf("[MAIN] Graphics load error: %d\n", gfxInitialized);
		return;
	}

	// Create the GPanels
	NNCreatorPanel* nnCreatorPanel =
		new NNCreatorPanel(serverInstance, "nnCreatorPanel", newWidth, newHeight);
	nnCreatorPanel->show(&cGfx);
	cGfx.addItem(nnCreatorPanel);

	// run gfx lib (blocks the main thread)
	cGfx.run();
}
