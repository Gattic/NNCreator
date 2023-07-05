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
#include "Backend/Database/ServiceData.h"
#include "Backend/Machine Learning/ImageInput.h"
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

class ML_Train : public GNet::Service
{
private:
	GNet::GServer* serverInstance;
	glades::NNetwork cNetwork;

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

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		class GNet::Connection* destination = data->getConnection();

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		shmea::GList cList = data->getList();

		if ((cList.size() == 1) && (cList.getString(0) == "KILL"))
		{
			if (!cNetwork.getRunning())
				return NULL;

			cNetwork.stop();
			printf("!!---KILLING NET---!!\n");
		}

		if (cList.size() < 3)
			return NULL;

		shmea::GString netName = cList.getString(0);
		shmea::GString testFName = cList.getString(1);
		int importType = cList.getInt(2);
		/*int64_t maxTimeStamp = cList.getLong(3);
		int64_t maxEpoch = cList.getLong(4);
		float maxAccuracy = cList.getFloat(5);*/
		// int64_t trainPct = cList.getLong(4), testPct = cList.getLong(5), validationPct =
		// cList.getLong(6);

		if (importType == glades::NNetwork::TYPE_CSV)
		{
			//
		}
		else if (importType == glades::NNetwork::TYPE_IMAGE)
		{
			// Load the images
			glades::ImageInput ii;
			ii.import(testFName.c_str());
			return NULL; // TODO DELETE THIS LINE
		}
		else if (importType == glades::NNetwork::TYPE_TEXT)
		{
			//
		}
		else
			return NULL;

		// Load the neural network
		if ((cNetwork.getEpochs() == 0) && (!cNetwork.load(netName.c_str())))
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
		shmea::GTable inputTable(testFName, ',', shmea::GTable::TYPE_FILE);
		glades::MetaNetwork* newTrainNet =
			glades::train(&cNetwork, inputTable, Arnold, serverInstance, destination);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new ML_Train(newInstance);
	}

	shmea::GString getName() const
	{
		return "ML_Train";
	}
};

#endif
