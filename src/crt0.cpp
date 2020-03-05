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
#include "crt0.h"
#include "Backend/Database/GList.h"
#include "Backend/Machine Learning/glades.h"
#include "Backend/Machine Learning/network.h"
#include "Backend/Networking/instance.h"
#include "Backend/Networking/main.h"
#include "Backend/Networking/service.h"
#include "Backend/Networking/socket.h"
#include "core/error.h"
#include "core/md5.h"
#include "core/version.h"
#include "main.h"
#include "services/ml_train.h"

bool NNCreator::running = true;
Version* NNCreator::version = new Version("0.55");
int NNCreator::debugType = DEBUG_SIMPLE;

/*!

*/
bool NNCreator::getRunning()
{
	return running;
}

Version NNCreator::getVersion()
{
	return *version;
}

int NNCreator::getDebugType()
{
	return debugType;
}

void NNCreator::stop()
{
	running = false;

	// shut down the server
	shutdown(GNet::getSockFD(), 2);
}

int main(int argc, char* argv[])
{
	// version & header
	printf("%s\n", NNCreator::getVersion().header().c_str());

	// For random numbers
	srand(time(NULL));

	// for machine learning initialization
	glades::init();

	// Add services
	ML_Train* ml_train_srvc = new ML_Train();
	addService(ml_train_srvc->getName(), ml_train_srvc);

	// command line args
	bool noguiMode = false;
	bool fullScreenMode = false;
	bool localOnly = false;
	if (argc > 1)
	{
		noguiMode = (strcmp(argv[1], "nogui") == 0);
		fullScreenMode = (strcmp(argv[1], "fullscreen") == 0);
		localOnly = (strcmp(argv[1], "local") == 0);
	}

	// Launch the server server
	pthread_t* commandThread = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_t* writerThread = (pthread_t*)malloc(sizeof(pthread_t));
	GNet::run(commandThread, writerThread, localOnly);

	// launch the gui
	if (!noguiMode)
		Frontend::run(fullScreenMode);

	// cleanup the glades
	glades::cleanup();

	// cleanup the networking threads
	GNet::stop();
	pthread_join(*commandThread, NULL);
	// pthread_join(*writerThread, NULL);
	free(commandThread);
	free(writerThread);

	// cleanup the networking
	GNet::Sockets::closeSockets();

	return 0;
}
