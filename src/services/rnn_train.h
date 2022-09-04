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
#ifndef _RNN_TRAIN
#define _RNN_TRAIN

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/glades.h"
#include "Backend/Machine Learning/metanetwork.h"
#include "Backend/Machine Learning/network.h"
#include "Backend/Networking/service.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/Graphics/graphics.h"

class NNInfo;

class RNN_Train : public GNet::Service
{
private:
	GNet::GServer* serverInstance;
	glades::NNetwork* cNetwork;

public:
	RNN_Train()
	{
		serverInstance = NULL;
		cNetwork = new glades::NNetwork(glades::NNetwork::TYPE_RNN);
	}

	RNN_Train(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
		cNetwork = new glades::NNetwork(glades::NNetwork::TYPE_RNN);
	}

	virtual ~RNN_Train()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		class GNet::Connection* destination = data->getConnection();

		printf("RNN WHY0\n");

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		printf("RNN WHY1\n");
		shmea::GList cList = data->getList();

		if ((cList.size() == 1) && (cList.getString(0) == "KILL"))
		{
			if (!cNetwork->getRunning())
				return NULL;

			cNetwork->stop();
			printf("!!---KILLING NET---!!\n");
		}

		printf("RNN WHY2\n");
		if (cList.size() < 3)
			return NULL;

		printf("RNN WHY3\n");
		shmea::GString netName = cList.getString(0);
		shmea::GString testFName = cList.getString(1);
		int importType = cList.getInt(2);
		/*int64_t maxTimeStamp = cList.getLong(3);
		int64_t maxEpoch = cList.getLong(4);
		float maxAccuracy = cList.getFloat(5);*/
		// int64_t trainPct = cList.getLong(4), testPct = cList.getLong(5), validationPct =
		// cList.getLong(6);

		// Load the neural network
		if ((cNetwork->getEpochs() == 0) && (!cNetwork->load(netName.c_str())))
		{
			printf("[NN] Unable to load \"%s\"", netName.c_str());
			return NULL;
		}

		// Termination Conditions
		glades::Terminator* Arnold = new glades::Terminator();
		/*Arnold->setTimestamp(maxTimeStamp);
		Arnold->setEpoch(maxEpoch);
		Arnold->setAccuracy(maxAccuracy);*/

		// Run the training and retrieve a metanetwork
		shmea::GTable inputTable(testFName, ',', importType);
		glades::MetaNetwork* newTrainNet =
			glades::train(cNetwork, inputTable, Arnold, serverInstance, destination);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new RNN_Train(newInstance);
	}

	shmea::GString getName() const
	{
		return "RNN_Train";
	}
};

#endif
