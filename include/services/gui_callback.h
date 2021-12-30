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
#ifndef _GUI_CALLBACK_PROG
#define _GUI_CALLBACK_PROG

#include "Backend/Database/ServiceData.h"
#include "Backend/Networking/service.h"

class GUI_Callback : public GNet::Service
{
private:

	GNet::GServer* serverInstance;
	GPanel* cPanel;

public:
	GUI_Callback()
	{
		serverInstance = NULL;
		cPanel = NULL;
	}

	GUI_Callback(GNet::GServer* newInstance, GPanel* newPanel)
	{
		serverInstance = newInstance;
		cPanel = newPanel;
	}

	~GUI_Callback()
	{
		serverInstance = NULL; // Not ours to delete
		cPanel = NULL;	   // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		if (!serverInstance)
			return NULL;

		if (!cPanel)
			return NULL;

		cPanel->addToQ(data);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new GUI_Callback(newInstance, cPanel);
	}

	shmea::GString getName() const
	{
		return "GUI_Callback";
	}
};

#endif
