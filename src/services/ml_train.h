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
#include "Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/DataObjects/TokenInput.h"
#include "Backend/Machine Learning/Networks/metanetwork.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Networking/service.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/Graphics/graphics.h"
#include <string>

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
			printf("\n!!---KILLING NET---!!\n");
		}

		if (cList.size() < 3)
			return NULL;

		shmea::GString modelName = cList.getString(0);
		shmea::GString inputFName = cList.getString(1);
		int inputType = cList.getInt(2);

		// Optional modern config payload:
		//  [3]=netType (int)
		//  [4]=lrScheduleType (int: 0 none, 1 step, 2 exp, 3 cosine)
		//  [5]=stepSize (int)
		//  [6]=gamma (float)
		//  [7]=tMax (int)
		//  [8]=minMult (float)
		//  [9]=globalGradClipNorm (float)
		//  [10]=perElementGradClip (float)
		//  [11]=tbpttWindowOverride (int)
		//  [12]=minibatchSizeOverride (int; 0 disables)
		//  [13]=optimizerType (int: 0 SGD_MOMENTUM, 1 ADAMW)
		//  [14]=adamBeta1 (float)
		//  [15]=adamBeta2 (float)
		//  [16]=adamEps (float)
		//  [17]=adamBiasCorrection (int: 0/1)
		//  [18]=tr.nHeadsOverride (int)
		//  [19]=tr.nKVHeadsOverride (int)
		//  [20]=tr.dFFOverride (int)
		//  [21]=tr.enableTokenEmbedding (int: 0/1)
		//  [22]=tr.vocabSizeOverride (int)
		//  [23]=tr.tieEmbeddings (int: 0/1)
		//  [24]=tr.padTokenId (int)
		//  [25]=tr.positionalEncoding (int: 0 none, 1 sin, 2 rope)
		//  [26]=tr.normType (int: 0 LN, 1 RMS)
		//  [27]=tr.ffnKind (int: 0 MLP, 1 SwiGLU)
		//  [28]=tr.ffnActivation (int: 0 ReLU, 1 GELU)
		//  [29]=tr.kvCacheDType (int: 0 F32, 1 F16, 2 BF16)
		//  [30]=tr.ropeDimOverride (int)
		//  [31]=tr.ropeTheta (float)
		//  [32]=tr.tokenLmLossKind (int: 0 full, 1 sampled)
		//  [33]=tr.tokenLmSampledNegatives (int)
		//
		// Backward compatibility:
		// - If [3] is a string, treat it as legacy weights filename (nn-state) and ignore modern fields.
		int netTypeOverride = -1;
		int schedType = 0;
		int stepSize = 0;
		float gamma = 1.0f;
		int tMax = 0;
		float minMult = 0.0f;
		float clipNorm = 0.0f;
		float perElemClip = 10.0f;
		int tbpttOverride = 0;
		int minibatchOverride = 0;

		int optimizerType = 0;
		float adamBeta1 = 0.9f;
		float adamBeta2 = 0.999f;
		float adamEps = 1e-8f;
		int adamBiasCorr = 1;

		int trHeads = 0;
		int trKVHeads = 0;
		int trDFF = 0;
		int trTokenEmbedding = 0;
		int trVocabOverride = 0;
		int trTieEmb = 1;
		int trPadTokenId = -1;
		int trPosEnc = 1;
		int trNorm = 0;
		int trFFNKind = 0;
		int trFFNAct = 0;
		int trKVCacheDType = 0;
		int trRoPEDim = 0;
		float trRoPETheta = 10000.0f;
		int trLossKind = 0;
		int trNeg = 64;
		// Legacy (pre-unified) weights filename (database/nn-state/<file>).
		// This service no longer loads these directly because NNetwork's internal graph state
		// is private; unified model packages should be used instead.
		shmea::GString legacyWeightsName = " ";
		bool hasLegacyWeights = false;

		int idx = 3;
		bool hasModernConfig = false;
		if (cList.size() > idx)
		{
			const shmea::GType t = cList[idx];
			if (t.getType() == shmea::GType::STRING_TYPE)
			{
				legacyWeightsName = cList.getString(idx);
				hasLegacyWeights = (legacyWeightsName != " ");
			}
			else
			{
				hasModernConfig = true;
				netTypeOverride = cList.getInt(idx);
				++idx;
				if (cList.size() > idx) { schedType = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { stepSize = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { gamma = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { tMax = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { minMult = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { clipNorm = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { perElemClip = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { tbpttOverride = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { minibatchOverride = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { optimizerType = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { adamBeta1 = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { adamBeta2 = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { adamEps = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { adamBiasCorr = cList.getInt(idx); ++idx; }

				if (cList.size() > idx) { trHeads = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trKVHeads = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trDFF = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trTokenEmbedding = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trVocabOverride = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trTieEmb = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trPadTokenId = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trPosEnc = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trNorm = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trFFNKind = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trFFNAct = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trKVCacheDType = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trRoPEDim = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trRoPETheta = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { trLossKind = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { trNeg = cList.getInt(idx); ++idx; }
			}
		}
		/*int64_t maxTimeStamp = cList.getLong(3);
		int64_t maxEpoch = cList.getLong(4);
		float maxAccuracy = cList.getFloat(5);*/
		// int64_t trainPct = cList.getLong(4), testPct = cList.getLong(5), validationPct =
		// cList.getLong(6);

		// Modify the paths to properly load the data later
		glades::DataInput* di = NULL;
		if (inputType == glades::DataInput::CSV)
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::NumberInput();
		}
		else if (inputType == glades::DataInput::IMAGE)
		{
			// inputFName = "datasets/images/" + inputFName + "/";
			di = new glades::ImageInput();
		}
		else if (inputType == glades::DataInput::TEXT)
		{
			inputFName = "datasets/" + inputFName;
			glades::TokenInput* ti = new glades::TokenInput();
			const int padId = (trTokenEmbedding ? trPadTokenId : -1);
			ti->setPadTokenId(padId);
			di = ti;
		}
		else
			return NULL;

		if (!di)
			return NULL;

		// Load the input data
		di->import(inputFName);

		// Load the model (unified model package).
		if (cNetwork.getEpochs() == 0)
		{
			const glades::NNetworkStatus st = cNetwork.loadModel(std::string(modelName.c_str()), di, netTypeOverride);
			if (!st.ok())
			{
				printf("[NN] Unable to load model \"%s\": %s\n", modelName.c_str(), st.message.c_str());
				delete di;
				return NULL;
			}
		}

		// Apply optional overrides (modern features).
		if (hasModernConfig)
		{
			glades::TrainingConfig cfg = cNetwork.getTrainingConfig();

			// Schedule
			if (schedType == 1)
				cfg.lrSchedule.setStep(stepSize, gamma);
			else if (schedType == 2)
				cfg.lrSchedule.setExp(gamma);
			else if (schedType == 3)
				cfg.lrSchedule.setCosine(tMax, minMult);
			else
				cfg.lrSchedule.setNone();

			// Core overrides
			cfg.globalGradClipNorm = clipNorm;
			cfg.perElementGradClip = perElemClip;
			cfg.tbpttWindowOverride = tbpttOverride;
			cfg.minibatchSizeOverride = minibatchOverride;

			// Optimizer
			if (optimizerType == 1)
			{
				cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
				cfg.optimizer.adamBeta1 = adamBeta1;
				cfg.optimizer.adamBeta2 = adamBeta2;
				cfg.optimizer.adamEps = adamEps;
				cfg.optimizer.adamBiasCorrection = (adamBiasCorr != 0);
			}
			else
			{
				cfg.optimizer.type = glades::OptimizerConfig::SGD_MOMENTUM;
			}

			// Transformer
			cfg.transformer.nHeadsOverride = trHeads;
			cfg.transformer.nKVHeadsOverride = trKVHeads;
			cfg.transformer.dFFOverride = trDFF;
			cfg.transformer.enableTokenEmbedding = (trTokenEmbedding != 0);
			cfg.transformer.vocabSizeOverride = trVocabOverride;
			cfg.transformer.tieEmbeddings = (trTieEmb != 0);
			cfg.transformer.padTokenId = trPadTokenId;
			cfg.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(trPosEnc);
			cfg.transformer.normType = static_cast<glades::TransformerRunConfig::NormType>(trNorm);
			cfg.transformer.ffnKind = static_cast<glades::TransformerRunConfig::FFNKind>(trFFNKind);
			cfg.transformer.ffnActivation = static_cast<glades::TransformerRunConfig::FFNActivationType>(trFFNAct);
			cfg.transformer.kvCacheDType = static_cast<glades::TransformerRunConfig::KVCacheDType>(trKVCacheDType);
			cfg.transformer.ropeDimOverride = trRoPEDim;
			cfg.transformer.ropeTheta = trRoPETheta;
			cfg.transformer.tokenLmLossKind = static_cast<glades::TransformerRunConfig::TokenLMLossKind>(trLossKind);
			cfg.transformer.tokenLmSampledNegatives = trNeg;

			const glades::NNetworkStatus stCfg = cNetwork.setTrainingConfig(cfg);
			if (!stCfg.ok())
				printf("[NN] warning: invalid training config overrides ignored: %s\n", stCfg.message.c_str());
		}

		// Legacy weights load (pre-unified persistence).
		// Intentionally not supported here anymore; these should be migrated into model packages.
		if (hasLegacyWeights)
			printf("[NN] note: legacy nn-state weights \"%s\" ignored; use unified model packages instead\n",
			       legacyWeightsName.c_str());

		// Termination Conditions
		/*cNetwork.terminator.setTimestamp(maxTimeStamp);
		cNetwork.terminator.setEpoch(maxEpoch);
		cNetwork.terminator.setAccuracy(maxAccuracy);*/

		// Run the training and retrieve a metanetwork
		(void)glades::train(&cNetwork, di, serverInstance, destination);

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
