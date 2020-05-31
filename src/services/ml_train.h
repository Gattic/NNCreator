// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#ifndef _ML_TRAIN
#define _ML_TRAIN

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/glades.h"
#include "Backend/Machine Learning/metanetwork.h"
#include "Backend/Machine Learning/network.h"
#include "Backend/Networking/service.h"
#include "Backend/Networking/socket.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/Graphics/graphics.h"

class NNInfo;

class ML_Train : public GNet::Service
{
private:
	GNet::GServer* serverInstance;

public:
	ML_Train()
	{
		serverInstance = NULL;
	}

	ML_Train(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~ML_Train()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::GList execute(class GNet::Connection* cConnection, const shmea::GList& data)
	{
		shmea::GList retList;
		if (data.size() < 3)
			return retList;

		std::string netName = data.getString(0);
		std::string testFName = data.getString(1);
		int importType = data.getInt(2);
		int64_t maxTimeStamp = data.getLong(3);
		int64_t maxEpoch = data.getLong(4);
		float maxAccuracy = data.getFloat(5);
		// int64_t trainPct = data.getLong(4), testPct = data.getLong(5), validationPct =
		// data.getLong(6);

		// load the neural network
		glades::NNetwork cNetwork;
		if (!cNetwork.load(netName))
		{
			printf("[NN] Unable to load \"%s\"", netName.c_str());
			return retList;
		}

		// Termination Conditions
		glades::Terminator* Arnold = new glades::Terminator();
		Arnold->setTimestamp(maxTimeStamp);
		Arnold->setEpoch(maxEpoch);
		Arnold->setAccuracy(maxAccuracy);

		// Run the training and retrieve a metanetwork
		shmea::GTable inputTable(testFName, ',', importType);
		glades::MetaNetwork* newTrainNet = glades::train(&cNetwork, inputTable, Arnold);

		return retList;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new ML_Train(newInstance);
	}

	std::string getName() const
	{
		return "ML_Train";
	}
};

#endif
