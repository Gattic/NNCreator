// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#ifndef _BAYES_TRAIN
#define _BAYES_TRAIN

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Machine Learning/bayes.h"
#include "Backend/Networking/service.h"

class Bayes_Train : public GNet::Service
{
private:
	GNet::GServer* serverInstance;
	glades::NNetwork cNetwork;

public:
	Bayes_Train()
	{
		serverInstance = NULL;
	}

	Bayes_Train(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~Bayes_Train()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		class GNet::Connection* destination = data->getConnection();

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		shmea::GList cList = data->getList();
		if (cList.size() < 3)
			return NULL;

		shmea::GString netName = cList.getString(0);

		shmea::GTable inputTable("datasets/btest.csv", ',', shmea::GTable::TYPE_FILE);

		glades::NaiveBayes bModel;
		shmea::GTable bTable = bModel.import(inputTable);
		bTable.print();
		bModel.train(bTable);

		// predict with model
		shmea::GList testList;
		testList.addString("silly");
		testList.addString("brown");
		int cls = bModel.predict(testList);
		printf("Predicted class %d\n", cls);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new Bayes_Train(newInstance);
	}

	shmea::GString getName() const
	{
		return "Bayes_Train";
	}
};

#endif
