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
#include "crt0.h"
#include "Backend/Database/GList.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Networking/main.h"
#include "Backend/Networking/service.h"
#include "Backend/Networking/socket.h"
#include "core/error.h"
#include "core/md5.h"
#include "core/version.h"
#include "main.h"
#include "services/bayes_train.h"
#include "services/ml_train.h"

bool NNCreator::running = true;
Version* NNCreator::version = new Version("0.58");
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
}

int main(int argc, char* argv[])
{
	// version & header
	printf("%s\n", NNCreator::getVersion().header().c_str());

	// For random numbers
	srand(time(NULL));

	// for machine learning initialization
	glades::init();

	GNet::GServer* serverInstance = new GNet::GServer();

	// Add services
	ML_Train* ml_train_srvc = new ML_Train(serverInstance);
	serverInstance->addService(ml_train_srvc);

	Bayes_Train* bayes_train_srvc = new Bayes_Train(serverInstance);
	serverInstance->addService(bayes_train_srvc);

	// command line args
	bool noguiMode = false;
	bool fullScreenMode = false;
	bool compatMode = false;
	bool localOnly = false;
	for (int i = 1; i < argc; ++i)
	{
		printf("Ingesting program paramter [%d]: %s\n", i, argv[i]);
		noguiMode = (strcmp(argv[i], "nogui") == 0);
		fullScreenMode = (strcmp(argv[i], "fullscreen") == 0);
		compatMode = (strcmp(argv[i], "fullscreen") == 0);
		localOnly = (strcmp(argv[i], "local") == 0);
	}

	// Launch the server server
	serverInstance->run("45024", localOnly);

	// Launch the gui
	if (!noguiMode)
		Frontend::run(serverInstance, fullScreenMode, compatMode);
	else
	{
		printf("Running in server mode\n");

		// Give time for the server to setup
		sleep(2);
		printf("Commands:\n");
		for (std::string line = ""; std::getline(std::cin, line);)
		{
			// printf("Input: %s\n", line.c_str());
			if ((!NNCreator::getRunning()) || (line == "exit") || (line == "quit"))
			{
				printf("Exiting...\n");
				NNCreator::stop();
				break;
			}
		}
	}

	// Cleanup GNet
	serverInstance->stop();

	return EXIT_SUCCESS;
}
