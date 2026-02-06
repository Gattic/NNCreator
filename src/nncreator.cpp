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
#include "nncreator.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/DataObjects/TokenInput.h"
#include "Backend/Machine Learning/GMath/gmath.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Networking/connection.h"
#include "Backend/Networking/main.h"
#include "Backend/Networking/service.h"
#include "Frontend/GFXUtilities/DrawNeuralNet.h"
#include "Frontend/GItems/GItem.h"
#include "Frontend/GLayouts/GLinearLayout.h"
#include "Frontend/GUI/RUCheckbox.h"
#include "Frontend/GUI/RUDropdown.h"
#include "Frontend/GUI/RUForm.h"
#include "Frontend/GUI/RUImageComponent.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/GUI/RUTabContainer.h"
#include "Frontend/GUI/RUTable.h"
#include "Frontend/GUI/Text/RUButton.h"
#include "Frontend/GUI/Text/RULabel.h"
#include "Frontend/GUI/Text/RUTextbox.h"
#include "Frontend/Graphics/graphics.h"
#include "Frontend/RUGraph/RUGraph.h"
#include "crt0.h"
#include "main.h"
#include "services/gui_callback.h"
#include <algorithm>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

// using namespace shmea;
using namespace glades;

namespace {
static bool isDir(const std::string& path)
{
	struct stat st;
	if (stat(path.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode);
}

static std::vector<std::string> listModelPackages()
{
	std::vector<std::string> out;
	const std::string root = "database/models";
	if (!isDir(root))
		return out;

	DIR* dir = opendir(root.c_str());
	if (!dir)
		return out;

	struct dirent* ent = NULL;
	while ((ent = readdir(dir)) != NULL)
	{
		const std::string name(ent->d_name ? ent->d_name : "");
		if (!name.size() || name[0] == '.')
			continue;

		// Only include directories (model packages).
		// Note: some filesystems may return DT_UNKNOWN; fall back to stat.
		bool isD = false;
		if (ent->d_type == DT_DIR)
			isD = true;
		else if (ent->d_type == DT_UNKNOWN)
			isD = isDir(root + "/" + name);
		if (!isD)
			continue;

		out.push_back(name);
	}

	closedir(dir);
	std::sort(out.begin(), out.end());
	return out;
}

static bool deleteRecursive(const std::string& path)
{
	struct stat st;
	if (lstat(path.c_str(), &st) != 0)
		return false;

	if (S_ISDIR(st.st_mode))
	{
		DIR* dir = opendir(path.c_str());
		if (!dir)
			return false;

		struct dirent* ent = NULL;
		while ((ent = readdir(dir)) != NULL)
		{
			const std::string name(ent->d_name ? ent->d_name : "");
			if (name == "." || name == "..")
				continue;
			if (!deleteRecursive(path + "/" + name))
			{
				closedir(dir);
				return false;
			}
		}
		closedir(dir);
		return (rmdir(path.c_str()) == 0);
	}

	return (unlink(path.c_str()) == 0);
}

static int scheduleTypeFromIndex(int idx)
{
	// maps to glades::LearningRateScheduleConfig::Type
	switch (idx)
	{
	case 1: return 1; // STEP
	case 2: return 2; // EXP
	case 3: return 3; // COSINE
	case 0:
	default: return 0; // NONE
	}
}
} // namespace

/*!
 * @brief NNCreatorPanel constructor
 * @details builds the NNCreator panel
 * @param name name of the panel
 * @param width width of the panel
 * @param height height of the panel
 */
NNCreatorPanel::NNCreatorPanel(const shmea::GString& name, int width, int height)
	: GPanel(name, width, height)
{
	serverInstance = NULL;
	netCount = 0;
	keepGraping = true;
	buildPanel();
}

NNCreatorPanel::NNCreatorPanel(GNet::GServer* newInstance, const shmea::GString& name, int width,
							   int height)
	: GPanel(name, width, height)
{
	serverInstance = newInstance;
	netCount = 0;
	keepGraping = true;
	buildPanel();
}

void NNCreatorPanel::buildPanel()
{
	lcMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	rocMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));

	pthread_mutex_init(lcMutex, NULL);
	pthread_mutex_init(rocMutex, NULL);

	// Add services
	GUI_Callback* gui_cb_srvc = new GUI_Callback(serverInstance, this);
	serverInstance->addService(gui_cb_srvc);

	currentHiddenLayerIndex = 0;
	InputLayerInfo* newInputLayer = new InputLayerInfo(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f);
	std::vector<HiddenLayerInfo*> newHiddenLayers;
	newHiddenLayers.push_back(new HiddenLayerInfo(2, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f));
	OutputLayerInfo* newOutputLayer = new OutputLayerInfo(1, OutputLayerInfo::REGRESSION);
	formInfo = new glades::NNInfo("", newInputLayer, newHiddenLayers, newOutputLayer);

	// Stores all the graphs
	GLinearLayout* graphsLayout = new GLinearLayout("graphsLayout");
	graphsLayout->setX(15);
	graphsLayout->setY(10);
	graphsLayout->setPadding(5);
	graphsLayout->setOrientation(GLinearLayout::VERTICAL);
	addSubItem(graphsLayout);

	// Top Row of graphs
	GLinearLayout* topGraphsLayout = new GLinearLayout("topGraphsLayout");
	topGraphsLayout->setPadding(5);
	topGraphsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	graphsLayout->addSubItem(topGraphsLayout);

	// Learning Curve Graph and Label
	GLinearLayout* lcGraphLayout = new GLinearLayout("lcGraphLayout");
	lcGraphLayout->setPadding(5);
	lcGraphLayout->setOrientation(GLinearLayout::VERTICAL);
	topGraphsLayout->addSubItem(lcGraphLayout);

	// Learning Curve Label
	RULabel* lblGraphLC = new RULabel();
	lblGraphLC->setText("Learning Curve (Epoch, Error)");
	lblGraphLC->setName("lblGraphLC");
	lcGraphLayout->addSubItem(lblGraphLC);

	// Learning curve graph
	lcGraph = new RUGraph(getWidth() / 4, getHeight() / 4, RUGraph::QUADRANTS_ONE);
	lcGraph->setName("lcGraph");
	lcGraphLayout->addSubItem(lcGraph);

	// ROC Curve Graph and Label
	GLinearLayout* rocGraphLayout = new GLinearLayout("rocGraphLayout");
	rocGraphLayout->setPadding(5);
	rocGraphLayout->setOrientation(GLinearLayout::VERTICAL);
	topGraphsLayout->addSubItem(rocGraphLayout);

	// ROC Curve Label
	RULabel* lblGraphROC = new RULabel();
	lblGraphROC->setText("ROC Curve (False Pos, True pos)");
	lblGraphROC->setName("lblGraphROC");
	rocGraphLayout->addSubItem(lblGraphROC);

	// ROC Curve Graph
	rocCurveGraph = new RUGraph(getWidth() / 4, getHeight() / 4,
								RUGraph::QUADRANTS_ONE); // -4 = adjustment to align with table
	rocCurveGraph->setName("rocCurveGraph");
	rocGraphLayout->addSubItem(rocCurveGraph);

	// Bottom row of graphs
	GLinearLayout* bottomGraphsLayout = new GLinearLayout("bottomGraphsLayout");
	bottomGraphsLayout->setPadding(5);
	bottomGraphsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	graphsLayout->addSubItem(bottomGraphsLayout);

	// Output Image and Label
	GLinearLayout* outputImageLayout = new GLinearLayout("outputImageLayout");
	outputImageLayout->setPadding(5);
	outputImageLayout->setOrientation(GLinearLayout::VERTICAL);
	bottomGraphsLayout->addSubItem(outputImageLayout);

	// Output Image Label
	RULabel* lblOutputImg = new RULabel();
	lblOutputImg->setText("Output Image");
	lblOutputImg->setName("lblOutputImg");
	outputImageLayout->addSubItem(lblOutputImg);

	shmea::GPointer<shmea::Image> newImage(new shmea::Image());
	// newImage->LoadPNG("resources/bg.png");

	// NN Output Image
	//	outputImage = new RUImageComponent();
	//	outputImage->setWidth(getWidth() / 4);
	//	outputImage->setHeight(getHeight() / 4);
	//	outputImage->setName("outputImage");
	//	outputImage->setBGImage(newImage);
	//	outputImageLayout->addSubItem(outputImage);
	neuralNetGraph = new RUGraph(getWidth() / 4, getHeight() / 4, RUGraph::QUADRANTS_ONE);
	neuralNetGraph->setName("neuralNetGraph");
	outputImageLayout->addSubItem(neuralNetGraph);

	// Confusion Table and Label
	GLinearLayout* confTableLayout = new GLinearLayout("confTableLayout");
	confTableLayout->setPadding(5);
	confTableLayout->setOrientation(GLinearLayout::VERTICAL);
	bottomGraphsLayout->addSubItem(confTableLayout);

	// Confusion Matrix Label
	RULabel* lblTableConf = new RULabel();
	lblTableConf->setText("Confusion Matrix");
	lblTableConf->setName("lblTableConf");
	confTableLayout->addSubItem(lblTableConf);

	// Confusion Matrix Table
	cMatrixTable = new RUTable();
	cMatrixTable->setRowsShown(5);
	cMatrixTable->setWidth(getWidth() / 4);
	cMatrixTable->setHeight(getHeight() / 4);
	cMatrixTable->setName("cMatrixTable");
	confTableLayout->addSubItem(cMatrixTable);

	//============LEFT============

	// Left Side forms
	GLinearLayout* leftSideLayout = new GLinearLayout("leftSideLayout");
	leftSideLayout->setX(15);
	leftSideLayout->setY(5 + bottomGraphsLayout->getY() + bottomGraphsLayout->getHeight());
	leftSideLayout->setPadding(10);
	leftSideLayout->setOrientation(GLinearLayout::VERTICAL);
	addSubItem(leftSideLayout);

	//============STATS============

	GLinearLayout* statsLayout = new GLinearLayout("statsLayout");
	statsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(statsLayout);

	// Epochs Label
	lblEpochs = new RULabel();
	lblEpochs->setText("");
	lblEpochs->setName("lblEpochs");
	statsLayout->addSubItem(lblEpochs);

	// Accuracy Label
	lblAccuracy = new RULabel();
	lblAccuracy->setText("");
	lblAccuracy->setName("lblAccuracy");
	statsLayout->addSubItem(lblAccuracy);

	//============FORM============

	// Neural Network Settings header
	lblSettings = new RULabel();
	lblSettings->setPadding(10);
	lblSettings->setText("Neural Network Settings");
	lblSettings->setName("lblSettings");
	leftSideLayout->addSubItem(lblSettings);

	// Tabs to keep settings usable on smaller windows.
	RUTabContainer* settingsTabs = new RUTabContainer();
	settingsTabs->setWidth(480);
	settingsTabs->setTabHeight(30);
	settingsTabs->setOptionsShown(3);
	settingsTabs->setPadding(6);
	settingsTabs->setName("settingsTabs");
	leftSideLayout->addSubItem(settingsTabs);
	settingsTabs->setSelectedTab(0);

	GLinearLayout* modelSettingsLayout = new GLinearLayout("modelSettingsLayout");
	modelSettingsLayout->setOrientation(GLinearLayout::VERTICAL);
	modelSettingsLayout->setPadding(6);
	settingsTabs->addTab("Model", modelSettingsLayout);

	GLinearLayout* trainingSettingsLayout = new GLinearLayout("trainingSettingsLayout");
	trainingSettingsLayout->setOrientation(GLinearLayout::VERTICAL);
	trainingSettingsLayout->setPadding(6);
	settingsTabs->addTab("Training", trainingSettingsLayout);

	GLinearLayout* dataRunSettingsLayout = new GLinearLayout("dataRunSettingsLayout");
	dataRunSettingsLayout->setOrientation(GLinearLayout::VERTICAL);
	dataRunSettingsLayout->setPadding(6);
	settingsTabs->addTab("Data/Run", dataRunSettingsLayout);

	GLinearLayout* loadNetLayout = new GLinearLayout("loadNetLayout");
	loadNetLayout->setOrientation(GLinearLayout::HORIZONTAL);
	modelSettingsLayout->addSubItem(loadNetLayout);

	// Neural Net selector label
	lblNeuralNet = new RULabel();
	lblNeuralNet->setText("Neural Network");
	lblNeuralNet->setName("lblNeuralNet");
	loadNetLayout->addSubItem(lblNeuralNet);

	// Neural Net selector
	ddNeuralNet = new RUDropdown();
	ddNeuralNet->setWidth(220);
	ddNeuralNet->setHeight(30);
	ddNeuralNet->setOptionsShown(3);
	ddNeuralNet->setName("ddNeuralNet");
	ddNeuralNet->setOptionChangedListener(
		GeneralListener(this, &NNCreatorPanel::nnSelectorChanged));
	loadNetLayout->addSubItem(ddNeuralNet);

	// Load button
	RUButton* btnLoad = new RUButton();
	btnLoad->setText("Reload List");
	btnLoad->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedLoad));
	btnLoad->setName("btnLoad");
	loadNetLayout->addSubItem(btnLoad);

	GLinearLayout* netNameLayout = new GLinearLayout("netNameLayout");
	netNameLayout->setOrientation(GLinearLayout::HORIZONTAL);
	modelSettingsLayout->addSubItem(netNameLayout);
	// Network Name label
	lblNetName = new RULabel();
	lblNetName->setX(6);
	lblNetName->setY(6 + lblNeuralNet->getY() + lblNeuralNet->getHeight());
	lblNetName->setText("Network Structure");
	lblNetName->setName("lblNetName");
	netNameLayout->addSubItem(lblNetName);

	// Network Name textbox
	tbNetName = new RUTextbox();
	tbNetName->setX(20 + lblNetName->getX() + lblNetName->getWidth());
	tbNetName->setY(lblNetName->getY());
	tbNetName->setWidth(220);
	tbNetName->setText("");
	tbNetName->setName("tbNetName");
	netNameLayout->addSubItem(tbNetName);

	// Save button
	btnSave = new RUButton("green");
	btnSave->setText("Save");
	btnSave->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedSave));
	btnSave->setName("btnSave");
	netNameLayout->addSubItem(btnSave);

	// Delete button
	btnDelete = new RUButton("red");
	btnDelete->setText("Delete");
	btnDelete->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedDelete));
	btnDelete->setName("btnDelete");
	netNameLayout->addSubItem(btnDelete);

	// ==== Modern training config (net type, LR schedule, grad clipping, TBPTT) ====
	GLinearLayout* netTypeLayout = new GLinearLayout("netTypeLayout");
	netTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	modelSettingsLayout->addSubItem(netTypeLayout);

	lblNetType = new RULabel();
	lblNetType->setWidth(200);
	lblNetType->setHeight(30);
	lblNetType->setText("Net Type");
	lblNetType->setName("lblNetType");
	netTypeLayout->addSubItem(lblNetType);

	ddNetType = new RUDropdown();
	ddNetType->setWidth(220);
	ddNetType->setHeight(30);
	ddNetType->setOptionsShown(6);
	ddNetType->setName("ddNetType");
	ddNetType->addOption("DFF");
	ddNetType->addOption("RNN");
	ddNetType->addOption("GRU");
	ddNetType->addOption("LSTM");
	ddNetType->addOption("Transformer (Enc)");
	ddNetType->addOption("Transformer (Dec)");
	ddNetType->setSelectedIndex(0);
	netTypeLayout->addSubItem(ddNetType);

	GLinearLayout* lrSchedLayout = new GLinearLayout("lrSchedLayout");
	lrSchedLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(lrSchedLayout);

	lblLRSchedule = new RULabel();
	lblLRSchedule->setWidth(200);
	lblLRSchedule->setHeight(30);
	lblLRSchedule->setText("LR Schedule");
	lblLRSchedule->setName("lblLRSchedule");
	lrSchedLayout->addSubItem(lblLRSchedule);

	ddLRSchedule = new RUDropdown();
	ddLRSchedule->setWidth(220);
	ddLRSchedule->setHeight(30);
	ddLRSchedule->setOptionsShown(4);
	ddLRSchedule->setName("ddLRSchedule");
	ddLRSchedule->addOption("None");
	ddLRSchedule->addOption("Step");
	ddLRSchedule->addOption("Exp");
	ddLRSchedule->addOption("Cosine");
	ddLRSchedule->setSelectedIndex(0);
	lrSchedLayout->addSubItem(ddLRSchedule);

	GLinearLayout* lrParams1 = new GLinearLayout("lrParams1");
	lrParams1->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(lrParams1);

	lblStepSize = new RULabel();
	lblStepSize->setWidth(120);
	lblStepSize->setHeight(30);
	lblStepSize->setText("Step");
	lblStepSize->setName("lblStepSize");
	lrParams1->addSubItem(lblStepSize);
	tbStepSize = new RUTextbox();
	tbStepSize->setWidth(80);
	tbStepSize->setHeight(30);
	tbStepSize->setText("3");
	tbStepSize->setName("tbStepSize");
	lrParams1->addSubItem(tbStepSize);

	lblGamma = new RULabel();
	lblGamma->setWidth(80);
	lblGamma->setHeight(30);
	lblGamma->setText("Gamma");
	lblGamma->setName("lblGamma");
	lrParams1->addSubItem(lblGamma);
	tbGamma = new RUTextbox();
	tbGamma->setWidth(120);
	tbGamma->setHeight(30);
	tbGamma->setText("0.25");
	tbGamma->setName("tbGamma");
	lrParams1->addSubItem(tbGamma);

	GLinearLayout* lrParams2 = new GLinearLayout("lrParams2");
	lrParams2->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(lrParams2);

	lblTMax = new RULabel();
	lblTMax->setWidth(120);
	lblTMax->setHeight(30);
	lblTMax->setText("TMax");
	lblTMax->setName("lblTMax");
	lrParams2->addSubItem(lblTMax);
	tbTMax = new RUTextbox();
	tbTMax->setWidth(80);
	tbTMax->setHeight(30);
	tbTMax->setText("50");
	tbTMax->setName("tbTMax");
	lrParams2->addSubItem(tbTMax);

	lblMinMult = new RULabel();
	lblMinMult->setWidth(80);
	lblMinMult->setHeight(30);
	lblMinMult->setText("Min");
	lblMinMult->setName("lblMinMult");
	lrParams2->addSubItem(lblMinMult);
	tbMinMult = new RUTextbox();
	tbMinMult->setWidth(120);
	tbMinMult->setHeight(30);
	tbMinMult->setText("0.0");
	tbMinMult->setName("tbMinMult");
	lrParams2->addSubItem(tbMinMult);

	GLinearLayout* clipLayout = new GLinearLayout("clipLayout");
	clipLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(clipLayout);

	lblGradClipNorm = new RULabel();
	lblGradClipNorm->setWidth(200);
	lblGradClipNorm->setHeight(30);
	lblGradClipNorm->setText("Grad Clip Norm");
	lblGradClipNorm->setName("lblGradClipNorm");
	clipLayout->addSubItem(lblGradClipNorm);
	tbGradClipNorm = new RUTextbox();
	tbGradClipNorm->setWidth(80);
	tbGradClipNorm->setHeight(30);
	tbGradClipNorm->setText("0.0");
	tbGradClipNorm->setName("tbGradClipNorm");
	clipLayout->addSubItem(tbGradClipNorm);

	lblPerElemClip = new RULabel();
	lblPerElemClip->setWidth(80);
	lblPerElemClip->setHeight(30);
	lblPerElemClip->setText("Elem");
	lblPerElemClip->setName("lblPerElemClip");
	clipLayout->addSubItem(lblPerElemClip);
	tbPerElemClip = new RUTextbox();
	tbPerElemClip->setWidth(120);
	tbPerElemClip->setHeight(30);
	tbPerElemClip->setText("10.0");
	tbPerElemClip->setName("tbPerElemClip");
	clipLayout->addSubItem(tbPerElemClip);

	GLinearLayout* tbpttLayout = new GLinearLayout("tbpttLayout");
	tbpttLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(tbpttLayout);

	lblTBPTT = new RULabel();
	lblTBPTT->setWidth(200);
	lblTBPTT->setHeight(30);
	lblTBPTT->setText("TBPTT Window");
	lblTBPTT->setName("lblTBPTT");
	tbpttLayout->addSubItem(lblTBPTT);
	tbTBPTT = new RUTextbox();
	tbTBPTT->setWidth(220);
	tbTBPTT->setHeight(30);
	tbTBPTT->setText("0");
	tbTBPTT->setName("tbTBPTT");
	tbpttLayout->addSubItem(tbTBPTT);

	// ==== TrainingConfig overrides (minibatch + optimizer + transformer knobs) ====
	GLinearLayout* minibatchOverrideLayout = new GLinearLayout("minibatchOverrideLayout");
	minibatchOverrideLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(minibatchOverrideLayout);

	lblMinibatchOverride = new RULabel();
	lblMinibatchOverride->setWidth(200);
	lblMinibatchOverride->setHeight(30);
	lblMinibatchOverride->setText("Minibatch Override");
	lblMinibatchOverride->setName("lblMinibatchOverride");
	minibatchOverrideLayout->addSubItem(lblMinibatchOverride);

	tbMinibatchOverride = new RUTextbox();
	tbMinibatchOverride->setWidth(220);
	tbMinibatchOverride->setHeight(30);
	tbMinibatchOverride->setText("0");
	tbMinibatchOverride->setName("tbMinibatchOverride");
	minibatchOverrideLayout->addSubItem(tbMinibatchOverride);

	GLinearLayout* optLayout = new GLinearLayout("optLayout");
	optLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(optLayout);

	lblOptimizer = new RULabel();
	lblOptimizer->setWidth(200);
	lblOptimizer->setHeight(30);
	lblOptimizer->setText("Optimizer");
	lblOptimizer->setName("lblOptimizer");
	optLayout->addSubItem(lblOptimizer);

	ddOptimizer = new RUDropdown();
	ddOptimizer->setWidth(220);
	ddOptimizer->setHeight(30);
	ddOptimizer->setOptionsShown(2);
	ddOptimizer->setName("ddOptimizer");
	ddOptimizer->addOption("SGD+Momentum");
	ddOptimizer->addOption("AdamW");
	ddOptimizer->setSelectedIndex(0);
	optLayout->addSubItem(ddOptimizer);

	GLinearLayout* adam1Layout = new GLinearLayout("adam1Layout");
	adam1Layout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(adam1Layout);

	lblAdamBeta1 = new RULabel();
	lblAdamBeta1->setWidth(120);
	lblAdamBeta1->setHeight(30);
	lblAdamBeta1->setText("Adam beta1");
	lblAdamBeta1->setName("lblAdamBeta1");
	adam1Layout->addSubItem(lblAdamBeta1);

	tbAdamBeta1 = new RUTextbox();
	tbAdamBeta1->setWidth(80);
	tbAdamBeta1->setHeight(30);
	tbAdamBeta1->setText("0.9");
	tbAdamBeta1->setName("tbAdamBeta1");
	adam1Layout->addSubItem(tbAdamBeta1);

	lblAdamBeta2 = new RULabel();
	lblAdamBeta2->setWidth(80);
	lblAdamBeta2->setHeight(30);
	lblAdamBeta2->setText("beta2");
	lblAdamBeta2->setName("lblAdamBeta2");
	adam1Layout->addSubItem(lblAdamBeta2);

	tbAdamBeta2 = new RUTextbox();
	tbAdamBeta2->setWidth(120);
	tbAdamBeta2->setHeight(30);
	tbAdamBeta2->setText("0.999");
	tbAdamBeta2->setName("tbAdamBeta2");
	adam1Layout->addSubItem(tbAdamBeta2);

	GLinearLayout* adam2Layout = new GLinearLayout("adam2Layout");
	adam2Layout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(adam2Layout);

	lblAdamEps = new RULabel();
	lblAdamEps->setWidth(120);
	lblAdamEps->setHeight(30);
	lblAdamEps->setText("Adam eps");
	lblAdamEps->setName("lblAdamEps");
	adam2Layout->addSubItem(lblAdamEps);

	tbAdamEps = new RUTextbox();
	tbAdamEps->setWidth(80);
	tbAdamEps->setHeight(30);
	tbAdamEps->setText("1e-8");
	tbAdamEps->setName("tbAdamEps");
	adam2Layout->addSubItem(tbAdamEps);

	chkAdamBiasCorrection = new RUCheckbox("Bias Corr");
	chkAdamBiasCorrection->setWidth(200);
	chkAdamBiasCorrection->setHeight(30);
	chkAdamBiasCorrection->setName("chkAdamBiasCorrection");
	chkAdamBiasCorrection->setCheck(true);
	adam2Layout->addSubItem(chkAdamBiasCorrection);

	// Transformer config block (always visible; only used for transformer net types)
	lblTransformerHeader = new RULabel();
	lblTransformerHeader->setPadding(6);
	lblTransformerHeader->setText("Transformer");
	lblTransformerHeader->setName("lblTransformerHeader");
	trainingSettingsLayout->addSubItem(lblTransformerHeader);

	GLinearLayout* trDimsLayout = new GLinearLayout("trDimsLayout");
	trDimsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trDimsLayout);

	lblTrHeads = new RULabel();
	lblTrHeads->setWidth(120);
	lblTrHeads->setHeight(30);
	lblTrHeads->setText("Heads");
	lblTrHeads->setName("lblTrHeads");
	trDimsLayout->addSubItem(lblTrHeads);
	tbTrHeads = new RUTextbox();
	tbTrHeads->setWidth(60);
	tbTrHeads->setHeight(30);
	tbTrHeads->setText("0");
	tbTrHeads->setName("tbTrHeads");
	trDimsLayout->addSubItem(tbTrHeads);

	lblTrKVHeads = new RULabel();
	lblTrKVHeads->setWidth(80);
	lblTrKVHeads->setHeight(30);
	lblTrKVHeads->setText("KV");
	lblTrKVHeads->setName("lblTrKVHeads");
	trDimsLayout->addSubItem(lblTrKVHeads);
	tbTrKVHeads = new RUTextbox();
	tbTrKVHeads->setWidth(60);
	tbTrKVHeads->setHeight(30);
	tbTrKVHeads->setText("0");
	tbTrKVHeads->setName("tbTrKVHeads");
	trDimsLayout->addSubItem(tbTrKVHeads);

	lblTrDFF = new RULabel();
	lblTrDFF->setWidth(60);
	lblTrDFF->setHeight(30);
	lblTrDFF->setText("dFF");
	lblTrDFF->setName("lblTrDFF");
	trDimsLayout->addSubItem(lblTrDFF);
	tbTrDFF = new RUTextbox();
	tbTrDFF->setWidth(80);
	tbTrDFF->setHeight(30);
	tbTrDFF->setText("0");
	tbTrDFF->setName("tbTrDFF");
	trDimsLayout->addSubItem(tbTrDFF);

	GLinearLayout* trTokLayout = new GLinearLayout("trTokLayout");
	trTokLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trTokLayout);

	chkTrTokenEmbedding = new RUCheckbox("Token Embedding");
	chkTrTokenEmbedding->setWidth(200);
	chkTrTokenEmbedding->setHeight(30);
	chkTrTokenEmbedding->setName("chkTrTokenEmbedding");
	chkTrTokenEmbedding->setCheck(false);
	trTokLayout->addSubItem(chkTrTokenEmbedding);

	lblTrVocabSize = new RULabel();
	lblTrVocabSize->setWidth(80);
	lblTrVocabSize->setHeight(30);
	lblTrVocabSize->setText("Vocab");
	lblTrVocabSize->setName("lblTrVocabSize");
	trTokLayout->addSubItem(lblTrVocabSize);
	tbTrVocabSize = new RUTextbox();
	tbTrVocabSize->setWidth(70);
	tbTrVocabSize->setHeight(30);
	tbTrVocabSize->setText("0");
	tbTrVocabSize->setName("tbTrVocabSize");
	trTokLayout->addSubItem(tbTrVocabSize);

	chkTrTieEmbeddings = new RUCheckbox("Tie");
	chkTrTieEmbeddings->setWidth(70);
	chkTrTieEmbeddings->setHeight(30);
	chkTrTieEmbeddings->setName("chkTrTieEmbeddings");
	chkTrTieEmbeddings->setCheck(true);
	trTokLayout->addSubItem(chkTrTieEmbeddings);

	lblTrPadTokenId = new RULabel();
	lblTrPadTokenId->setWidth(50);
	lblTrPadTokenId->setHeight(30);
	lblTrPadTokenId->setText("Pad");
	lblTrPadTokenId->setName("lblTrPadTokenId");
	trTokLayout->addSubItem(lblTrPadTokenId);
	tbTrPadTokenId = new RUTextbox();
	tbTrPadTokenId->setWidth(60);
	tbTrPadTokenId->setHeight(30);
	tbTrPadTokenId->setText("-1");
	tbTrPadTokenId->setName("tbTrPadTokenId");
	trTokLayout->addSubItem(tbTrPadTokenId);

	GLinearLayout* trStyleLayout = new GLinearLayout("trStyleLayout");
	trStyleLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trStyleLayout);

	lblTrPosEnc = new RULabel();
	lblTrPosEnc->setWidth(120);
	lblTrPosEnc->setHeight(30);
	lblTrPosEnc->setText("PosEnc");
	lblTrPosEnc->setName("lblTrPosEnc");
	trStyleLayout->addSubItem(lblTrPosEnc);
	ddTrPosEnc = new RUDropdown();
	ddTrPosEnc->setWidth(120);
	ddTrPosEnc->setHeight(30);
	ddTrPosEnc->setOptionsShown(3);
	ddTrPosEnc->setName("ddTrPosEnc");
	ddTrPosEnc->addOption("None");
	ddTrPosEnc->addOption("Sin");
	ddTrPosEnc->addOption("RoPE");
	ddTrPosEnc->setSelectedIndex(1);
	trStyleLayout->addSubItem(ddTrPosEnc);

	lblTrNorm = new RULabel();
	lblTrNorm->setWidth(60);
	lblTrNorm->setHeight(30);
	lblTrNorm->setText("Norm");
	lblTrNorm->setName("lblTrNorm");
	trStyleLayout->addSubItem(lblTrNorm);
	ddTrNorm = new RUDropdown();
	ddTrNorm->setWidth(100);
	ddTrNorm->setHeight(30);
	ddTrNorm->setOptionsShown(2);
	ddTrNorm->setName("ddTrNorm");
	ddTrNorm->addOption("LN");
	ddTrNorm->addOption("RMS");
	ddTrNorm->setSelectedIndex(0);
	trStyleLayout->addSubItem(ddTrNorm);

	GLinearLayout* trFfnLayout = new GLinearLayout("trFfnLayout");
	trFfnLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trFfnLayout);

	lblTrFFNKind = new RULabel();
	lblTrFFNKind->setWidth(120);
	lblTrFFNKind->setHeight(30);
	lblTrFFNKind->setText("FFN");
	lblTrFFNKind->setName("lblTrFFNKind");
	trFfnLayout->addSubItem(lblTrFFNKind);
	ddTrFFNKind = new RUDropdown();
	ddTrFFNKind->setWidth(120);
	ddTrFFNKind->setHeight(30);
	ddTrFFNKind->setOptionsShown(2);
	ddTrFFNKind->setName("ddTrFFNKind");
	ddTrFFNKind->addOption("MLP");
	ddTrFFNKind->addOption("SwiGLU");
	ddTrFFNKind->setSelectedIndex(0);
	trFfnLayout->addSubItem(ddTrFFNKind);

	lblTrFFNAct = new RULabel();
	lblTrFFNAct->setWidth(60);
	lblTrFFNAct->setHeight(30);
	lblTrFFNAct->setText("Act");
	lblTrFFNAct->setName("lblTrFFNAct");
	trFfnLayout->addSubItem(lblTrFFNAct);
	ddTrFFNAct = new RUDropdown();
	ddTrFFNAct->setWidth(100);
	ddTrFFNAct->setHeight(30);
	ddTrFFNAct->setOptionsShown(2);
	ddTrFFNAct->setName("ddTrFFNAct");
	ddTrFFNAct->addOption("ReLU");
	ddTrFFNAct->addOption("GELU");
	ddTrFFNAct->setSelectedIndex(0);
	trFfnLayout->addSubItem(ddTrFFNAct);

	GLinearLayout* trRopeLayout = new GLinearLayout("trRopeLayout");
	trRopeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trRopeLayout);

	lblTrRoPEDim = new RULabel();
	lblTrRoPEDim->setWidth(120);
	lblTrRoPEDim->setHeight(30);
	lblTrRoPEDim->setText("RoPE Dim");
	lblTrRoPEDim->setName("lblTrRoPEDim");
	trRopeLayout->addSubItem(lblTrRoPEDim);
	tbTrRoPEDim = new RUTextbox();
	tbTrRoPEDim->setWidth(60);
	tbTrRoPEDim->setHeight(30);
	tbTrRoPEDim->setText("0");
	tbTrRoPEDim->setName("tbTrRoPEDim");
	trRopeLayout->addSubItem(tbTrRoPEDim);

	lblTrRoPETheta = new RULabel();
	lblTrRoPETheta->setWidth(80);
	lblTrRoPETheta->setHeight(30);
	lblTrRoPETheta->setText("Theta");
	lblTrRoPETheta->setName("lblTrRoPETheta");
	trRopeLayout->addSubItem(lblTrRoPETheta);
	tbTrRoPETheta = new RUTextbox();
	tbTrRoPETheta->setWidth(120);
	tbTrRoPETheta->setHeight(30);
	tbTrRoPETheta->setText("10000");
	tbTrRoPETheta->setName("tbTrRoPETheta");
	trRopeLayout->addSubItem(tbTrRoPETheta);

	GLinearLayout* trLossLayout = new GLinearLayout("trLossLayout");
	trLossLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trLossLayout);

	lblTrLossKind = new RULabel();
	lblTrLossKind->setWidth(120);
	lblTrLossKind->setHeight(30);
	lblTrLossKind->setText("LM Loss");
	lblTrLossKind->setName("lblTrLossKind");
	trLossLayout->addSubItem(lblTrLossKind);
	ddTrLossKind = new RUDropdown();
	ddTrLossKind->setWidth(140);
	ddTrLossKind->setHeight(30);
	ddTrLossKind->setOptionsShown(2);
	ddTrLossKind->setName("ddTrLossKind");
	ddTrLossKind->addOption("Full");
	ddTrLossKind->addOption("Sampled");
	ddTrLossKind->setSelectedIndex(0);
	trLossLayout->addSubItem(ddTrLossKind);

	lblTrNegSamples = new RULabel();
	lblTrNegSamples->setWidth(60);
	lblTrNegSamples->setHeight(30);
	lblTrNegSamples->setText("Neg");
	lblTrNegSamples->setName("lblTrNegSamples");
	trLossLayout->addSubItem(lblTrNegSamples);
	tbTrNegSamples = new RUTextbox();
	tbTrNegSamples->setWidth(80);
	tbTrNegSamples->setHeight(30);
	tbTrNegSamples->setText("64");
	tbTrNegSamples->setName("tbTrNegSamples");
	trLossLayout->addSubItem(tbTrNegSamples);

	GLinearLayout* trKvLayout = new GLinearLayout("trKvLayout");
	trKvLayout->setOrientation(GLinearLayout::HORIZONTAL);
	trainingSettingsLayout->addSubItem(trKvLayout);

	lblTrKVCacheDType = new RULabel();
	lblTrKVCacheDType->setWidth(120);
	lblTrKVCacheDType->setHeight(30);
	lblTrKVCacheDType->setText("KV Cache");
	lblTrKVCacheDType->setName("lblTrKVCacheDType");
	trKvLayout->addSubItem(lblTrKVCacheDType);
	ddTrKVCacheDType = new RUDropdown();
	ddTrKVCacheDType->setWidth(220);
	ddTrKVCacheDType->setHeight(30);
	ddTrKVCacheDType->setOptionsShown(3);
	ddTrKVCacheDType->setName("ddTrKVCacheDType");
	ddTrKVCacheDType->addOption("F32");
	ddTrKVCacheDType->addOption("F16");
	ddTrKVCacheDType->addOption("BF16");
	ddTrKVCacheDType->setSelectedIndex(0);
	trKvLayout->addSubItem(ddTrKVCacheDType);

	// Input/Output Data Type Layout
	GLinearLayout* dataTypeLayout = new GLinearLayout("dataTypeLayout");
	dataTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	dataRunSettingsLayout->addSubItem(dataTypeLayout);

	// Input/Output Data Type label
	RULabel* lbldatarow = new RULabel();
	lbldatarow->setText("Data Type/Path");
	lbldatarow->setName("lbldatarow");
	dataTypeLayout->addSubItem(lbldatarow);

	// Input/Output Data Type
	ddDataType = new RUDropdown();
	ddDataType->setWidth(160);
	ddDataType->setHeight(30);
	ddDataType->setOptionsShown(3);
	ddDataType->setName("ddDataType");
	ddDataType->setOptionChangedListener(
		GeneralListener(this, &NNCreatorPanel::clickedDSTypeSwitch));
	dataTypeLayout->addSubItem(ddDataType);

	ddDataType->addOption("CSV");
	ddDataType->addOption("Image");
	ddDataType->addOption("Text");

	//  sample path textbox
	ddDatasets = new RUDropdown();
	ddDatasets->setWidth(220);
	ddDatasets->setHeight(30);
	ddDatasets->setOptionsShown(3);
	ddDatasets->setName("ddDatasets");
	dataTypeLayout->addSubItem(ddDatasets);

	// Run layout
	GLinearLayout* runTestLayout = new GLinearLayout("runTestLayout");
	runTestLayout->setOrientation(GLinearLayout::HORIZONTAL);
	dataRunSettingsLayout->addSubItem(runTestLayout);

	// Run Button
	RUButton* sendButton = new RUButton("green");
	sendButton->setText("Run");
	sendButton->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedRun));
	sendButton->setName("sendButton");
	runTestLayout->addSubItem(sendButton);

	// Continue Button
	RUButton* contButton = new RUButton("blue");
	contButton->setText("Continue");
	contButton->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedContinue));
	contButton->setName("contButton");
	runTestLayout->addSubItem(contButton);

	// Kill Button
	RUButton* killButton = new RUButton("red");
	killButton->setText("Kill");
	killButton->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedKill));
	killButton->setName("killButton");
	runTestLayout->addSubItem(killButton);

	// Cross Validation Layout
	GLinearLayout* crossValLayout = new GLinearLayout("crossValLayout");
	crossValLayout->setOrientation(GLinearLayout::HORIZONTAL);
	// dataRunSettingsLayout->addSubItem(crossValLayout); // TODO uncomment when cross val is tested

	// Cross Validation checkbox
	chkCrossVal = new RUCheckbox("Cross Validate");
	chkCrossVal->setWidth(200);
	chkCrossVal->setHeight(30);
	chkCrossVal->setName("chkCrossVal");
	chkCrossVal->setCheck(true);
	chkCrossVal->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::checkedCV));
	crossValLayout->addSubItem(chkCrossVal);

	// train % label
	lblttv = new RULabel();
	lblttv->setText("Train/Test/Val");
	lblttv->setName("lblttv");
	crossValLayout->addSubItem(lblttv);

	// train %
	tbTrainPct = new RUTextbox();
	tbTrainPct->setWidth(80);
	tbTrainPct->setText("70");
	tbTrainPct->setName("tbTrainPct");
	crossValLayout->addSubItem(tbTrainPct);

	// test %
	tbTestPct = new RUTextbox();
	tbTestPct->setWidth(80);
	tbTestPct->setText("20");
	tbTestPct->setName("tbTestPct");
	crossValLayout->addSubItem(tbTestPct);

	// validation %
	tbValidationPct = new RUTextbox();
	tbValidationPct->setWidth(80);
	tbValidationPct->setText("10");
	tbValidationPct->setName("tbValidationPct");
	crossValLayout->addSubItem(tbValidationPct);

	//============RIGHT============

	// Right Side forms
	GLinearLayout* rightSideLayout = new GLinearLayout("rightSideLayout");
	rightSideLayout->setX(getWidth() - 500);
	rightSideLayout->setY(45);
	rightSideLayout->setOrientation(GLinearLayout::VERTICAL);
	rightSideLayout->setPadding(4);
	addSubItem(rightSideLayout);

	// Layer Tab navigation
	layerTabs = new RUTabContainer();
	layerTabs->setWidth(120);
	layerTabs->setTabHeight(30);
	layerTabs->setOptionsShown(3);
	layerTabs->setPadding(10);
	layerTabs->setName("layerTabs");
	rightSideLayout->addSubItem(layerTabs);
	layerTabs->setSelectedTab(1);

	inputOverallLayout = new GLinearLayout("inputOverallLayout");
	inputOverallLayout->setX(getWidth() - 500);
	inputOverallLayout->setY(90);
	inputOverallLayout->setOrientation(GLinearLayout::VERTICAL);
	layerTabs->addTab("Input", inputOverallLayout);

	// Edit Input Layer Header
	lblEditInputLayer = new RULabel();
	lblEditInputLayer->setText("Edit Input Layer");
	lblEditInputLayer->setName("lblEditInputLayer");
	inputOverallLayout->addSubItem(lblEditInputLayer);

	RUForm* inputLayerForm = new RUForm("inputLayerForm");
	inputOverallLayout->addSubItem(inputLayerForm);

	GLinearLayout* batchUpdateLayout = new GLinearLayout("batchUpdateLayout");
	batchUpdateLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(batchUpdateLayout);

	// Batch label
	RULabel* lblBatchSize = new RULabel();
	lblBatchSize->setText("Batch Size ");
	lblBatchSize->setName("lblBatchSize");
	batchUpdateLayout->addSubItem(lblBatchSize);

	// Batch
	tbBatchSize = new RUTextbox();
	tbBatchSize->setWidth(160);
	tbBatchSize->setText("1");
	tbBatchSize->setName("tbBatchSize");
	batchUpdateLayout->addSubItem(tbBatchSize);
	inputLayerForm->addSubItem(tbBatchSize);

	GLinearLayout* inputLCLayout = new GLinearLayout("inputLCLayout");
	inputLCLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputLCLayout);

	// learning rate label
	lblinputLR = new RULabel();
	lblinputLR->setText("Learning Rate");
	lblinputLR->setName("lblinputLR");
	inputLCLayout->addSubItem(lblinputLR);

	// learning rate textbox
	tbinputLR = new RUTextbox();
	tbinputLR->setWidth(160);
	tbinputLR->setName("tbinputLR");
	inputLCLayout->addSubItem(tbinputLR);
	inputLayerForm->addSubItem(tbinputLR);

	GLinearLayout* inputMFLayout = new GLinearLayout("inputMFLayout");
	inputMFLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputMFLayout);

	// momentum factor label
	lblinputMF = new RULabel();
	lblinputMF->setText("Momentum Factor");
	lblinputMF->setName("lblinputMF");
	inputMFLayout->addSubItem(lblinputMF);

	// momentum factor textbox
	tbinputMF = new RUTextbox();
	tbinputMF->setWidth(160);
	tbinputMF->setName("tbinputMF");
	inputMFLayout->addSubItem(tbinputMF);
	inputLayerForm->addSubItem(tbinputMF);

	GLinearLayout* inputWDLayout1 = new GLinearLayout("inputWDLayout1");
	inputWDLayout1->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputWDLayout1);

	// weight decay label
	lblinputWD1 = new RULabel();
	lblinputWD1->setText("L1 Regularization");
	lblinputWD1->setName("lblinputWD1");
	inputWDLayout1->addSubItem(lblinputWD1);

	// weight decay textbox
	tbinputWD1 = new RUTextbox();
	tbinputWD1->setWidth(160);
	tbinputWD1->setName("tbinputWD1");
	inputWDLayout1->addSubItem(tbinputWD1);
	inputLayerForm->addSubItem(tbinputWD1);

	GLinearLayout* inputWDLayout2 = new GLinearLayout("inputWDLayout2");
	inputWDLayout2->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputWDLayout2);

	// weight decay label
	lblinputWD2 = new RULabel();
	lblinputWD2->setText("L2 Regularization");
	lblinputWD2->setName("lblinputWD2");
	inputWDLayout2->addSubItem(lblinputWD2);

	// weight decay textbox
	tbinputWD2 = new RUTextbox();
	tbinputWD2->setWidth(160);
	tbinputWD2->setName("tbinputWD2");
	inputWDLayout2->addSubItem(tbinputWD2);
	inputLayerForm->addSubItem(tbinputWD2);

	GLinearLayout* inputDropoutLayout = new GLinearLayout("inputDropoutLayout");
	inputDropoutLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputDropoutLayout);

	// pHidden label
	lblinputDropout = new RULabel();
	lblinputDropout->setText("Dropout p: ");
	lblinputDropout->setName("lblinputDropout");
	inputDropoutLayout->addSubItem(lblinputDropout);

	// pHidden textbox
	tbinputDropout = new RUTextbox();
	tbinputDropout->setWidth(160);
	tbinputDropout->setText("0.0");
	tbinputDropout->setName("tbinputDropout");
	inputDropoutLayout->addSubItem(tbinputDropout);
	inputLayerForm->addSubItem(tbinputDropout);

	GLinearLayout* inputATLayout = new GLinearLayout("inputATLayout");
	inputATLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputATLayout);

	// activation functions label
	lblinputAF = new RULabel();
	lblinputAF->setText("Activation Type");
	lblinputAF->setName("lblinputAF");
	inputATLayout->addSubItem(lblinputAF);

	// activation functions textbox
	ddinputAF = new RUDropdown();
	ddinputAF->setWidth(160);
	ddinputAF->setHeight(30);
	ddinputAF->setOptionsShown(3);
	ddinputAF->setName("ddinputAF");
	inputATLayout->addSubItem(ddinputAF);

	ddinputAF->addOption("Tanh");
	ddinputAF->addOption("PWise Tanh");
	ddinputAF->addOption("Sigmoid");
	ddinputAF->addOption("PWise Sigmoid");
	ddinputAF->addOption("Linear");
	ddinputAF->addOption("ReLu");
	ddinputAF->addOption("Leaky ReLu");

	GLinearLayout* inputAPLayout = new GLinearLayout("inputAPLayout");
	inputAPLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(inputAPLayout);

	// Activation param label
	RULabel* lblinputAP = new RULabel();
	lblinputAP->setText("Activation Param");
	lblinputAP->setName("lblinputAP");
	inputAPLayout->addSubItem(lblinputAP);

	// Activation param textbox
	tbinputAP = new RUTextbox();
	tbinputAP->setWidth(160);
	tbinputAP->setName("tbinputAP");
	inputAPLayout->addSubItem(tbinputAP);
	inputLayerForm->addSubItem(tbinputAP);

	//-----------

	// Preview label Header
	RULabel* lblPreview = new RULabel();
	lblPreview->setPadding(10);
	lblPreview->setText("Preview");
	lblPreview->setName("lblPreview");
	inputOverallLayout->addSubItem(lblPreview);

	GLinearLayout* previewButtonsLayout = new GLinearLayout("previewButtonsLayout");
	previewButtonsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	inputOverallLayout->addSubItem(previewButtonsLayout);

	// Preview Train Data Button
	RUButton* btnPreviewTrain = new RUButton("blue");
	btnPreviewTrain->setText("Training Data");
	btnPreviewTrain->setMouseDownListener(
		GeneralListener(this, &NNCreatorPanel::clickedPreviewTrain));
	btnPreviewTrain->setName("btnPreviewTrain");
	previewButtonsLayout->addSubItem(btnPreviewTrain);

	// Preview Test Data Button
	RUButton* btnPreviewTest = new RUButton("blue");
	btnPreviewTest->setText("Testing Data");
	btnPreviewTest->setMouseDownListener(
		GeneralListener(this, &NNCreatorPanel::clickedPreviewTest));
	btnPreviewTest->setName("btnPreviewTest");
	previewButtonsLayout->addSubItem(btnPreviewTest);

	previewTabs = new RUTabContainer();
	previewTabs->setWidth(90);
	previewTabs->setTabHeight(30);
	previewTabs->setTabsVisible(false);
	previewTabs->setOptionsShown(3);
	previewTabs->setPadding(10);
	previewTabs->setName("previewTabs");
	inputOverallLayout->addSubItem(previewTabs);
	previewTabs->setSelectedTab(1);

	// Preview Table
	previewTable = new RUTable();
	previewTable->setRowsShown(8);
	previewTable->setWidth(getWidth() / 4);
	previewTable->setHeight(getHeight() / 4);
	previewTable->setName("previewTable");
	previewTabs->addTab("CSV", previewTable);

	previewImageLayout = new GLinearLayout("previewImageLayout");
	previewImageLayout->setOrientation(GLinearLayout::VERTICAL);
	previewTabs->addTab("Image", previewImageLayout);

	shmea::GPointer<shmea::Image> prevImage(new shmea::Image());
	// prevImage->LoadPNG("resources/bg.png");

	// Preview Image
	previewImage = new RUImageComponent();
	previewImage->setWidth(getWidth() / 4);
	previewImage->setHeight(getHeight() / 4);
	previewImage->setName("previewImage");
	previewImage->setBGImage(prevImage);
	previewImageLayout->addSubItem(previewImage);

	GLinearLayout* prevImageBtnsLayout = new GLinearLayout("prevImageBtnsLayout");
	prevImageBtnsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	previewImageLayout->addSubItem(prevImageBtnsLayout);

	// Previous Button
	RUButton* btnPrevious = new RUButton("blue");
	btnPrevious->setText("Previous");
	btnPrevious->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedPrevious));
	btnPrevious->setName("btnPrevious");
	prevImageBtnsLayout->addSubItem(btnPrevious);

	// Next Button
	RUButton* btnNext = new RUButton("blue");
	btnNext->setText("Next");
	btnNext->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedNext));
	btnNext->setName("btnNext");
	prevImageBtnsLayout->addSubItem(btnNext);

	hiddenOverallLayout = new GLinearLayout("hiddenOverallLayout");
	hiddenOverallLayout->setX(getWidth() - 500);
	hiddenOverallLayout->setY(90);
	hiddenOverallLayout->setOrientation(GLinearLayout::VERTICAL);
	layerTabs->addTab("Hidden", hiddenOverallLayout);

	RUForm* hiddenLayerForm = new RUForm("hiddenLayerForm");
	hiddenOverallLayout->addSubItem(hiddenLayerForm);

	// Hidden Layer Title Label
	RULabel* lbllayertitle = new RULabel();
	lbllayertitle->setText("Edit Hidden Layers");
	lbllayertitle->setName("lbllayertitle");
	hiddenOverallLayout->addSubItem(lbllayertitle);

	GLinearLayout* hlcLayout = new GLinearLayout("hlcLayout");
	hlcLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(hlcLayout);

	// hidden layer count label
	lblHiddenLayerCount = new RULabel();
	lblHiddenLayerCount->setPadding(10);
	lblHiddenLayerCount->setText("Hidden Layer Count");
	lblHiddenLayerCount->setName("lblHiddenLayerCount");
	hlcLayout->addSubItem(lblHiddenLayerCount);

	// hidden layer count textbox
	tbHiddenLayerCount = new RUTextbox();
	tbHiddenLayerCount->setWidth(160);
	tbHiddenLayerCount->setText(shmea::GString::intTOstring(formInfo->numHiddenLayers()));
	tbHiddenLayerCount->setName("tbHiddenLayerCount");
	tbHiddenLayerCount->setLoseFocusListener(GeneralListener(this, &NNCreatorPanel::tbHLLoseFocus));
	hlcLayout->addSubItem(tbHiddenLayerCount);
	hiddenLayerForm->addSubItem(tbHiddenLayerCount);

	GLinearLayout* hlSelectLayout = new GLinearLayout("hlSelectLayout");
	hlSelectLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(hlSelectLayout);

	// edit hidden layer header
	lblEditHiddenLayer = new RULabel();
	lblEditHiddenLayer->setText("Edit Hidden Layer");
	lblEditHiddenLayer->setName("lblEditHiddenLayer");
	hlSelectLayout->addSubItem(lblEditHiddenLayer);

	// index to edit dropdown
	ddIndexToEdit = new RUDropdown();
	ddIndexToEdit->setWidth(160);
	ddIndexToEdit->setHeight(30);
	ddIndexToEdit->setOptionsShown(3);
	ddIndexToEdit->setName("ddIndexToEdit");
	ddIndexToEdit->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedEditSwitch));
	hlSelectLayout->addSubItem(ddIndexToEdit);

	GLinearLayout* hlSizeLayout = new GLinearLayout("hlSizeLayout");
	hlSizeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(hlSizeLayout);

	// hidden layer size label
	lblHiddenLayerSize = new RULabel();
	lblHiddenLayerSize->setText("Size");
	lblHiddenLayerSize->setName("lblHiddenLayerSize");
	hlSizeLayout->addSubItem(lblHiddenLayerSize);

	// hidden layer size textbox
	tbHiddenLayerSize = new RUTextbox();
	tbHiddenLayerSize->setWidth(160);
	tbHiddenLayerSize->setName("tbHiddenLayerSize");
	hlSizeLayout->addSubItem(tbHiddenLayerSize);
	hiddenLayerForm->addSubItem(tbHiddenLayerSize);

	GLinearLayout* lcLayout = new GLinearLayout("lcLayout");
	lcLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(lcLayout);

	// learning rate label
	lblLearningRate = new RULabel();
	lblLearningRate->setText("Learning Rate");
	lblLearningRate->setName("lblLearningRate");
	lcLayout->addSubItem(lblLearningRate);

	// learning rate textbox
	tbLearningRate = new RUTextbox();
	tbLearningRate->setWidth(160);
	tbLearningRate->setName("tbLearningRate");
	lcLayout->addSubItem(tbLearningRate);
	hiddenLayerForm->addSubItem(tbLearningRate);

	GLinearLayout* mfLayout = new GLinearLayout("mfLayout");
	mfLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(mfLayout);

	// momentum factor label
	lblMomentumFactor = new RULabel();
	lblMomentumFactor->setText("Momentum Factor");
	lblMomentumFactor->setName("lblMomentumFactor");
	mfLayout->addSubItem(lblMomentumFactor);

	// momentum factor textbox
	tbMomentumFactor = new RUTextbox();
	tbMomentumFactor->setWidth(160);
	tbMomentumFactor->setName("tbMomentumFactor");
	mfLayout->addSubItem(tbMomentumFactor);
	hiddenLayerForm->addSubItem(tbMomentumFactor);

	GLinearLayout* wdLayout1 = new GLinearLayout("wdLayout1");
	wdLayout1->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(wdLayout1);

	// weight decay label
	lblWeightDecay1 = new RULabel();
	lblWeightDecay1->setText("L1 Regularization");
	lblWeightDecay1->setName("lblWeightDecay1");
	wdLayout1->addSubItem(lblWeightDecay1);

	// weight decay textbox
	tbWeightDecay1 = new RUTextbox();
	tbWeightDecay1->setWidth(160);
	tbWeightDecay1->setName("tbWeightDecay1");
	wdLayout1->addSubItem(tbWeightDecay1);
	hiddenLayerForm->addSubItem(tbWeightDecay1);

	GLinearLayout* wdLayout2 = new GLinearLayout("wdLayout2");
	wdLayout2->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(wdLayout2);

	// weight decay label
	lblWeightDecay2 = new RULabel();
	lblWeightDecay2->setText("L2 Regularization");
	lblWeightDecay2->setName("lblWeightDecay2");
	wdLayout2->addSubItem(lblWeightDecay2);

	// weight decay textbox
	tbWeightDecay2 = new RUTextbox();
	tbWeightDecay2->setWidth(160);
	tbWeightDecay2->setName("tbWeightDecay2");
	wdLayout2->addSubItem(tbWeightDecay2);
	hiddenLayerForm->addSubItem(tbWeightDecay2);

	GLinearLayout* pHiddenLayout = new GLinearLayout("pHiddenLayout");
	pHiddenLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(pHiddenLayout);

	// pHidden label
	lblPHidden = new RULabel();
	lblPHidden->setText("Hidden Layer p: ");
	lblPHidden->setName("lblPHidden");
	pHiddenLayout->addSubItem(lblPHidden);

	// pHidden textbox
	tbPHidden = new RUTextbox();
	tbPHidden->setWidth(160);
	tbPHidden->setName("tbPHidden");
	pHiddenLayout->addSubItem(tbPHidden);
	hiddenLayerForm->addSubItem(tbPHidden);

	GLinearLayout* actTypeLayout = new GLinearLayout("actTypeLayout");
	actTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(actTypeLayout);

	// activation functions label
	lblActivationFunctions = new RULabel();
	lblActivationFunctions->setText("Activation Type");
	lblActivationFunctions->setName("lblActivationFunctions");
	actTypeLayout->addSubItem(lblActivationFunctions);

	// activation functions textbox
	ddActivationFunctions = new RUDropdown();
	ddActivationFunctions->setWidth(160);
	ddActivationFunctions->setHeight(30);
	ddActivationFunctions->setOptionsShown(3);
	ddActivationFunctions->setName("ddActivationFunctions");
	actTypeLayout->addSubItem(ddActivationFunctions);

	ddActivationFunctions->addOption("Tanh");
	ddActivationFunctions->addOption("PWise Tanh");
	ddActivationFunctions->addOption("Sigmoid");
	ddActivationFunctions->addOption("PWise Sigmoid");
	ddActivationFunctions->addOption("Linear");
	ddActivationFunctions->addOption("ReLu");
	ddActivationFunctions->addOption("Leaky ReLu");

	GLinearLayout* actParamLayout = new GLinearLayout("actParamLayout");
	actParamLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(actParamLayout);

	// Activation param label
	RULabel* lblActivationParam = new RULabel();
	lblActivationParam->setText("Activation Param");
	lblActivationParam->setName("lblActivationParam");
	actParamLayout->addSubItem(lblActivationParam);

	// Activation param textbox
	tbActivationParam = new RUTextbox();
	tbActivationParam->setWidth(160);
	tbActivationParam->setName("tbActivationParam");
	actParamLayout->addSubItem(tbActivationParam);
	hiddenLayerForm->addSubItem(tbActivationParam);

	GLinearLayout* hCopyLayout = new GLinearLayout("hCopyLayout");
	hCopyLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(hCopyLayout);

	// Layer Dest label
	RULabel* lblCopyDest = new RULabel();
	lblCopyDest->setText("Dest Layer");
	lblCopyDest->setName("lblCopyDest");
	hCopyLayout->addSubItem(lblCopyDest);

	// copy/remove destination textbox
	tbCopyDestination = new RUTextbox();
	tbCopyDestination->setWidth(160);
	tbCopyDestination->setName("tbCopyDestination");
	hCopyLayout->addSubItem(tbCopyDestination);

	GLinearLayout* bCopyLayout = new GLinearLayout("bCopyLayout");
	bCopyLayout->setOrientation(GLinearLayout::HORIZONTAL);
	hiddenOverallLayout->addSubItem(bCopyLayout);

	// Copy button
	RUButton* btnCopy = new RUButton();
	btnCopy->setText("Copy");
	btnCopy->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedCopy));
	btnCopy->setName("btnCopy");
	bCopyLayout->addSubItem(btnCopy);

	// Remove button
	RUButton* btnRemove = new RUButton("red");
	btnRemove->setText("Remove");
	btnRemove->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedRemove));
	btnRemove->setName("btnRemove");
	bCopyLayout->addSubItem(btnRemove);

	outputOverallLayout = new GLinearLayout("outputOverallLayout");
	outputOverallLayout->setX(getWidth() - 500);
	outputOverallLayout->setY(90);
	outputOverallLayout->setOrientation(GLinearLayout::VERTICAL);
	layerTabs->addTab("Output", outputOverallLayout);

	// Edit Output Layer Header
	lblEditOutputLayer = new RULabel();
	lblEditOutputLayer->setText("Edit Output Layer");
	lblEditOutputLayer->setName("lblEditOutputLayer");
	outputOverallLayout->addSubItem(lblEditOutputLayer);

	GLinearLayout* outputTypeLayout = new GLinearLayout("outputTypeLayout");
	outputTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	outputOverallLayout->addSubItem(outputTypeLayout);

	// output type label
	lblOutputType = new RULabel();
	lblOutputType->setText("Output Type");
	lblOutputType->setName("lblOutputType");
	outputTypeLayout->addSubItem(lblOutputType);

	// output type dropdown
	ddOutputType = new RUDropdown();
	ddOutputType->setWidth(210);
	ddOutputType->setHeight(30);
	ddOutputType->setOptionsShown(2);
	ddOutputType->setName("ddOutputType");

	ddOutputType->addOption("Regression");
	ddOutputType->addOption("Classification");
	ddOutputType->addOption("KL Divergence");
	outputTypeLayout->addSubItem(ddOutputType);

	GLinearLayout* outputSizeLayout = new GLinearLayout("outputSizeLayout");
	outputSizeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	outputOverallLayout->addSubItem(outputSizeLayout);

	// output layer size label
	lblOutputLayerSize = new RULabel();
	lblOutputLayerSize->setText("Size");
	lblOutputLayerSize->setName("lblOutputLayerSize");
	outputSizeLayout->addSubItem(lblOutputLayerSize);

	// output layer size textbox
	tbOutputLayerSize = new RUTextbox();
	tbOutputLayerSize->setWidth(210);
	tbOutputLayerSize->setText("1");
	tbOutputLayerSize->setName("tbOutputLayerSize");
	outputSizeLayout->addSubItem(tbOutputLayerSize);

	//============END OF RIGHT============

	loadDDNN();
	populateIndexToEdit();
	populateInputLayerForm();
	populateHLayerForm();

	loadDatasets();
}

void NNCreatorPanel::onStart()
{
	// printf("RUComponentContainer Size: %ld\n", ruComponentContainer.size());
}

NNCreatorPanel::~NNCreatorPanel()
{
	pthread_mutex_destroy(lcMutex);
	if (lcMutex)
		free(lcMutex);

	pthread_mutex_destroy(rocMutex);
	if (rocMutex)
		free(rocMutex);
}

void NNCreatorPanel::loadDDNN()
{
	// clear the old items
	ddNeuralNet->clearOptions();

	// add the "new" item option
	ddNeuralNet->addOption("New");

	// add items from unified model packages to dropdown
	const std::vector<std::string> models = listModelPackages();
	for (size_t i = 0; i < models.size(); ++i)
		ddNeuralNet->addOption(models[i].c_str());
}

/*!
 * @brief input layer textbox populator
 * @details fills in the Input Layer variable textboxes and dropdowns
 */
void NNCreatorPanel::populateInputLayerForm()
{
	InputLayerInfo* inputLayer = formInfo->getInputLayer();

	tbinputLR->setText(shmea::GString::floatTOstring(inputLayer->getLearningRate()));
	tbinputMF->setText(shmea::GString::floatTOstring(inputLayer->getMomentumFactor()));
	tbinputWD1->setText(shmea::GString::floatTOstring(inputLayer->getWeightDecay1()));
	tbinputWD2->setText(shmea::GString::floatTOstring(inputLayer->getWeightDecay2()));
	tbinputDropout->setText(shmea::GString::floatTOstring(inputLayer->getPDropout()));
	tbinputAP->setText(shmea::GString::floatTOstring(inputLayer->getActivationParam()));
	ddinputAF->setSelectedIndex(inputLayer->getActivationType());
}

/*!
 * @brief hidden layer textbox populator
 * @details fills in the Hidden Layer variable textboxes and dropdowns based on
 * the
 * currentHiddenLayerIndex and hiddenLayers variables
 */
void NNCreatorPanel::populateHLayerForm()
{
	if (currentHiddenLayerIndex < 0 || currentHiddenLayerIndex >= formInfo->numHiddenLayers())
		return;

	HiddenLayerInfo* currentLayer = formInfo->getLayers()[currentHiddenLayerIndex];
	ddIndexToEdit->setSelectedIndex(currentHiddenLayerIndex);
	tbHiddenLayerSize->setText(shmea::GString::intTOstring(currentLayer->size()));

	tbLearningRate->setText(shmea::GString::floatTOstring(currentLayer->getLearningRate()));
	tbMomentumFactor->setText(shmea::GString::floatTOstring(currentLayer->getMomentumFactor()));
	tbWeightDecay1->setText(shmea::GString::floatTOstring(currentLayer->getWeightDecay1()));
	tbWeightDecay2->setText(shmea::GString::floatTOstring(currentLayer->getWeightDecay2()));
	tbPHidden->setText(shmea::GString::floatTOstring(currentLayer->getPDropout()));
	tbActivationParam->setText(shmea::GString::floatTOstring(currentLayer->getActivationParam()));
	ddActivationFunctions->setSelectedIndex(currentLayer->getActivationType());
}

/*!
 * @brief store hidden layer textbox values
 * @details stores the Hidden Layer Variable textbox values into
 * hiddenLayers[currentHiddenLayerIndex]
 */
void NNCreatorPanel::syncFormVar()
{
	// Input Stuff
	InputLayerInfo* inputLayer = formInfo->getInputLayer();

	shmea::GType batchSize =
		shmea::GString::Typify(tbBatchSize->getText(), tbBatchSize->getText().size());
	formInfo->setBatchSize(batchSize.getLong());

	shmea::GType inputLR =
		shmea::GString::Typify(tbinputLR->getText(), tbinputLR->getText().size());
	inputLayer->setLearningRate(inputLR.getFloat());

	shmea::GType inputMF =
		shmea::GString::Typify(tbinputMF->getText(), tbinputMF->getText().size());
	inputLayer->setMomentumFactor(inputMF.getFloat());

	shmea::GType inputWD1 =
		shmea::GString::Typify(tbinputWD1->getText(), tbinputWD1->getText().size());
	inputLayer->setWeightDecay1(inputWD1.getFloat());

	shmea::GType inputWD2 =
		shmea::GString::Typify(tbinputWD2->getText(), tbinputWD2->getText().size());
	inputLayer->setWeightDecay2(inputWD2.getFloat());

	shmea::GType inputDropout =
		shmea::GString::Typify(tbinputDropout->getText(), tbinputDropout->getText().size());
	inputLayer->setPDropout(inputDropout.getFloat());

	int inputAT = ddinputAF->getSelectedIndex();
	switch (inputAT)
	{
	case 0: {
		inputLayer->setActivationType(GMath::TANH);
		break;
	}
	case 1: {
		inputLayer->setActivationType(GMath::TANHP);
		break;
	}
	case 2: {
		inputLayer->setActivationType(GMath::SIGMOID);
		break;
	}
	case 3: {
		inputLayer->setActivationType(GMath::SIGMOIDP);
		break;
	}
	case 4: {
		inputLayer->setActivationType(GMath::LINEAR);
		break;
	}
	case 5: {
		inputLayer->setActivationType(GMath::RELU);
		break;
	}
	case 6: {
		inputLayer->setActivationType(GMath::LEAKY);
		break;
	}
	default: {
		inputLayer->setActivationType(GMath::TANH);
		break;
	}
	}

	shmea::GType inputAP =
		shmea::GString::Typify(tbinputAP->getText(), tbinputAP->getText().size());
	inputLayer->setActivationParam(inputAP.getFloat());

	// Output Stuff

	shmea::GType outputSize =
		shmea::GString::Typify(tbOutputLayerSize->getText(), tbOutputLayerSize->getText().size());
	formInfo->setOutputSize(outputSize.getFloat());

	shmea::GType outputType((int)ddOutputType->getSelectedIndex()); // this is probably fine...
	formInfo->setOutputType(outputType.getLong());

	if (currentHiddenLayerIndex < 0 || currentHiddenLayerIndex >= formInfo->numHiddenLayers())
		return;

	HiddenLayerInfo* currentLayer = formInfo->getLayers()[currentHiddenLayerIndex];

	shmea::GType pHidden =
		shmea::GString::Typify(tbPHidden->getText(), tbPHidden->getText().size());
	currentLayer->setPDropout(pHidden.getFloat());

	shmea::GType hLayerSize =
		shmea::GString::Typify(tbHiddenLayerSize->getText(), tbHiddenLayerSize->getText().size());
	currentLayer->setSize((int)hLayerSize.getLong());

	shmea::GType learningRate =
		shmea::GString::Typify(tbLearningRate->getText(), tbLearningRate->getText().size());
	currentLayer->setLearningRate(learningRate.getFloat());

	shmea::GType momentumFactor =
		shmea::GString::Typify(tbMomentumFactor->getText(), tbMomentumFactor->getText().size());
	currentLayer->setMomentumFactor(momentumFactor.getFloat());

	shmea::GType weightDecay1 =
		shmea::GString::Typify(tbWeightDecay1->getText(), tbWeightDecay1->getText().size());
	currentLayer->setWeightDecay1(weightDecay1.getFloat());

	shmea::GType weightDecay2 =
		shmea::GString::Typify(tbWeightDecay2->getText(), tbWeightDecay2->getText().size());
	currentLayer->setWeightDecay2(weightDecay2.getFloat());

	int activationType = ddActivationFunctions->getSelectedIndex();
	switch (activationType)
	{
	case 0: {
		currentLayer->setActivationType(GMath::TANH);
		break;
	}
	case 1: {
		currentLayer->setActivationType(GMath::TANHP);
		break;
	}
	case 2: {
		currentLayer->setActivationType(GMath::SIGMOID);
		break;
	}
	case 3: {
		currentLayer->setActivationType(GMath::SIGMOIDP);
		break;
	}
	case 4: {
		currentLayer->setActivationType(GMath::LINEAR);
		break;
	}
	case 5: {
		currentLayer->setActivationType(GMath::RELU);
		break;
	}
	case 6: {
		currentLayer->setActivationType(GMath::LEAKY);
		break;
	}
	default: {
		currentLayer->setActivationType(GMath::TANH);
		break;
	}
	}

	shmea::GType activationParam =
		shmea::GString::Typify(tbActivationParam->getText(), tbActivationParam->getText().size());
	currentLayer->setActivationParam(activationParam.getFloat());
}
/*!
 * @brief IndexToEdit populator
 * @details populates the IndexToEdit dropdown, usually whenever the Hidden
 * Layer Count textbox
 * changes
 */
void NNCreatorPanel::populateIndexToEdit(int newSelectedIndex)
{
	ddIndexToEdit->clearOptions();

	// needed to offset the clearOptions action
	for (int i = 0; i < formInfo->numHiddenLayers(); ++i)
		ddIndexToEdit->addOption(shmea::GString::intTOstring(i));

	if (newSelectedIndex < formInfo->numHiddenLayers())
		ddIndexToEdit->setSelectedIndex(newSelectedIndex);
}

/*!
 * @brief load neural net
 * @details load neural net variables into the form
 * @param skeleton neural network architecture
 */
void NNCreatorPanel::loadNNet(glades::NNInfo* info)
{
	formInfo = info;
	shmea::GString netName = formInfo->getName().c_str();
	tbNetName->setText(netName);

	float pInput = formInfo->getPInput();
	tbinputDropout->setText(shmea::GString::floatTOstring(pInput));

	int batchSize = formInfo->getBatchSize();
	tbBatchSize->setText(shmea::GString::intTOstring(batchSize));

	currentHiddenLayerIndex = 0;
	tbHiddenLayerCount->setText(shmea::GString::intTOstring(formInfo->numHiddenLayers()));

	int outputSize = formInfo->getOutputLayerSize();
	int outputType = formInfo->getOutputType();
	tbOutputLayerSize->setText(shmea::GString::intTOstring(outputSize));
	ddOutputType->setSelectedIndex(outputType);

	populateInputLayerForm();
	populateHLayerForm();
	populateIndexToEdit(currentHiddenLayerIndex);

	// Display a popup alert
	shmea::GString msgBoxText = "Loaded \"" + netName + "\"";
	RUMsgBox::MsgBox(this, "Neural Net", msgBoxText, RUMsgBox::MESSAGEBOX);
}

/*!
 * @brief parse Cross Val percentage parser
 * @details parses a percentage into a percent value
 * @param spct the string containing the percentage
 * @return the percentage between 0 and 100
 */
int64_t NNCreatorPanel::parsePct(const shmea::GType& spct)
{
	shmea::GType pct(spct);
	if (pct.getType() != shmea::GType::LONG_TYPE)
		return -1l;

	int64_t pctVal = pct.getLong();
	if (pctVal < 0 || pctVal > 100)
		return -1l;

	return pctVal;
}

void NNCreatorPanel::loadDatasets()
{
	if (!ddDataType)
		return;

	if (!ddDatasets)
		return;

	ddDatasets->clearOptions();
	trainingRowIndex = 0;
	testingRowIndex = 0;
	prevImageFlag = 0;

	int dataType = 0;
	shmea::GString folderName = "datasets/";
	if (ddDataType->getSelectedText() == "CSV")
	{
		dataType = 0;
		previewTable->setVisible(true);
		previewImageLayout->setVisible(false);
	}
	else if (ddDataType->getSelectedText() == "Image")
	{
		dataType = 1;
		folderName += "images/";
		previewTable->setVisible(false);
		previewImageLayout->setVisible(true);
	}
	else if (ddDataType->getSelectedText() == "Text")
	{
		dataType = 2;
		previewTable->setVisible(false);
		previewImageLayout->setVisible(false);
	}
	else
		return;

	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(folderName.c_str())) == NULL)
	{
		printf("[ML] -%s\n", folderName.c_str());
		return;
	}

	// loop through the directory
	while ((ent = readdir(dir)) != NULL)
	{
		// don't want the current directory, parent or hidden files/folders
		shmea::GString fname(ent->d_name);
		if (fname[0] == '.')
			continue;

		if ((ent->d_type == DT_DIR) && (dataType == 1))
		{
			// printf("Folder[%d]: %s \n", ent->d_type, fname.c_str());
			ddDatasets->addOption(fname);
		}
		else if ((ent->d_type == DT_REG) && (dataType == 0 || dataType == 2))
		{
			// printf("File[%d]: %s \n", ent->d_type, fname.c_str());
			ddDatasets->addOption(fname);
		}
		else
			continue;
	}

	closedir(dir);
	ddDatasets->setOptionsShown(3);
}

/*!
 * @brief Save NN
 * @details save a neural network using the Save_NN network service
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::clickedSave(const shmea::GString& cmpName, int x, int y)
{
	if (!tbNetName)
		return;

	if (tbNetName->getText().length() == 0)
		return;

	// make sure the layers are up to date
	syncFormVar();

	const shmea::GString modelName = tbNetName->getText();
	formInfo->setName(modelName.c_str());

	const int netType = (ddNetType ? ddNetType->getSelectedIndex() : glades::NNetwork::TYPE_DFF);
	glades::NNetwork net(formInfo, netType);

	// Apply modern training config (will be persisted into the model manifest).
	glades::TrainingConfig cfg = net.getTrainingConfig();

	const int schedIdx = (ddLRSchedule ? ddLRSchedule->getSelectedIndex() : 0);
	const int schedType = scheduleTypeFromIndex(schedIdx);

	const int stepSize = (tbStepSize ? shmea::GString::Typify(tbStepSize->getText(), tbStepSize->getText().size()).getInt() : 0);
	const float gamma = (tbGamma ? shmea::GString::Typify(tbGamma->getText(), tbGamma->getText().size()).getFloat() : 1.0f);
	const int tMax = (tbTMax ? shmea::GString::Typify(tbTMax->getText(), tbTMax->getText().size()).getInt() : 0);
	const float minMult = (tbMinMult ? shmea::GString::Typify(tbMinMult->getText(), tbMinMult->getText().size()).getFloat() : 0.0f);

	const float clipNorm = (tbGradClipNorm ? shmea::GString::Typify(tbGradClipNorm->getText(), tbGradClipNorm->getText().size()).getFloat() : 0.0f);
	const float perElem = (tbPerElemClip ? shmea::GString::Typify(tbPerElemClip->getText(), tbPerElemClip->getText().size()).getFloat() : 10.0f);
	const int tbpttOverride = (tbTBPTT ? shmea::GString::Typify(tbTBPTT->getText(), tbTBPTT->getText().size()).getInt() : 0);
	const int minibatchOverride = (tbMinibatchOverride ? shmea::GString::Typify(tbMinibatchOverride->getText(), tbMinibatchOverride->getText().size()).getInt() : 0);

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
	cfg.perElementGradClip = perElem;
	cfg.tbpttWindowOverride = tbpttOverride;
	cfg.minibatchSizeOverride = minibatchOverride;

	// Optimizer
	const int optIdx = (ddOptimizer ? ddOptimizer->getSelectedIndex() : 0);
	if (optIdx == 1)
	{
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		cfg.optimizer.adamBeta1 = (tbAdamBeta1 ? shmea::GString::Typify(tbAdamBeta1->getText(), tbAdamBeta1->getText().size()).getFloat() : 0.9f);
		cfg.optimizer.adamBeta2 = (tbAdamBeta2 ? shmea::GString::Typify(tbAdamBeta2->getText(), tbAdamBeta2->getText().size()).getFloat() : 0.999f);
		cfg.optimizer.adamEps = (tbAdamEps ? shmea::GString::Typify(tbAdamEps->getText(), tbAdamEps->getText().size()).getFloat() : 1e-8f);
		cfg.optimizer.adamBiasCorrection = (chkAdamBiasCorrection ? chkAdamBiasCorrection->isChecked() : true);
	}
	else
	{
		cfg.optimizer.type = glades::OptimizerConfig::SGD_MOMENTUM;
	}

	// Transformer knobs (used only for transformer net types; safe to always populate)
	cfg.transformer.nHeadsOverride = (tbTrHeads ? shmea::GString::Typify(tbTrHeads->getText(), tbTrHeads->getText().size()).getInt() : 0);
	cfg.transformer.nKVHeadsOverride = (tbTrKVHeads ? shmea::GString::Typify(tbTrKVHeads->getText(), tbTrKVHeads->getText().size()).getInt() : 0);
	cfg.transformer.dFFOverride = (tbTrDFF ? shmea::GString::Typify(tbTrDFF->getText(), tbTrDFF->getText().size()).getInt() : 0);

	cfg.transformer.enableTokenEmbedding = (chkTrTokenEmbedding ? chkTrTokenEmbedding->isChecked() : false);
	cfg.transformer.vocabSizeOverride = (tbTrVocabSize ? shmea::GString::Typify(tbTrVocabSize->getText(), tbTrVocabSize->getText().size()).getInt() : 0);
	cfg.transformer.tieEmbeddings = (chkTrTieEmbeddings ? chkTrTieEmbeddings->isChecked() : true);
	cfg.transformer.padTokenId = (tbTrPadTokenId ? shmea::GString::Typify(tbTrPadTokenId->getText(), tbTrPadTokenId->getText().size()).getInt() : -1);

	cfg.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(ddTrPosEnc ? ddTrPosEnc->getSelectedIndex() : glades::TransformerRunConfig::POSENC_SINUSOIDAL);
	cfg.transformer.normType = static_cast<glades::TransformerRunConfig::NormType>(ddTrNorm ? ddTrNorm->getSelectedIndex() : glades::TransformerRunConfig::NORM_LAYERNORM);
	cfg.transformer.ffnKind = static_cast<glades::TransformerRunConfig::FFNKind>(ddTrFFNKind ? ddTrFFNKind->getSelectedIndex() : glades::TransformerRunConfig::FFN_MLP);
	cfg.transformer.ffnActivation = static_cast<glades::TransformerRunConfig::FFNActivationType>(ddTrFFNAct ? ddTrFFNAct->getSelectedIndex() : glades::TransformerRunConfig::FFN_RELU);
	cfg.transformer.kvCacheDType = static_cast<glades::TransformerRunConfig::KVCacheDType>(ddTrKVCacheDType ? ddTrKVCacheDType->getSelectedIndex() : glades::TransformerRunConfig::KV_CACHE_F32);

	cfg.transformer.ropeDimOverride = (tbTrRoPEDim ? shmea::GString::Typify(tbTrRoPEDim->getText(), tbTrRoPEDim->getText().size()).getInt() : 0);
	cfg.transformer.ropeTheta = (tbTrRoPETheta ? shmea::GString::Typify(tbTrRoPETheta->getText(), tbTrRoPETheta->getText().size()).getFloat() : 10000.0f);

	cfg.transformer.tokenLmLossKind = static_cast<glades::TransformerRunConfig::TokenLMLossKind>(ddTrLossKind ? ddTrLossKind->getSelectedIndex() : glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
	cfg.transformer.tokenLmSampledNegatives = (tbTrNegSamples ? shmea::GString::Typify(tbTrNegSamples->getText(), tbTrNegSamples->getText().size()).getInt() : 64);

	// Apply config as a single validated update.
	{
		const glades::NNetworkStatus stCfg = net.setTrainingConfig(cfg);
		if (!stCfg.ok())
		{
			const shmea::GString msg = "Invalid training config: " + shmea::GString(stCfg.message.c_str());
			RUMsgBox::MsgBox(this, "Model Package", msg, RUMsgBox::MESSAGEBOX);
			return;
		}
	}

	// Best-effort: initialize the model's weight tensors based on the current dataset selection.
	// If no dataset is selected, saveModel() may still succeed depending on implementation.
	glades::DataInput* di = NULL;
	if (ddDataType && ddDatasets && ddDatasets->getSelectedText().length() > 0)
	{
		shmea::GString inputFName = ddDatasets->getSelectedText();
		const shmea::GString dtype = ddDataType->getSelectedText();
		if (dtype == "CSV")
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::NumberInput();
		}
		else if (dtype == "Image")
		{
			di = new glades::ImageInput();
		}
		else if (dtype == "Text")
		{
			inputFName = "datasets/" + inputFName;
			glades::TokenInput* ti = new glades::TokenInput();
			// If token embedding mode is enabled, use padTokenId for next-token targets when provided.
			// Otherwise leave it disabled (<0) so sequence-final targets are omitted.
			const int padId = (cfg.transformer.enableTokenEmbedding ? cfg.transformer.padTokenId : -1);
			ti->setPadTokenId(padId);
			di = ti;
		}

		if (di)
		{
			di->import(inputFName);
			// A test pass is enough to force graph/tensor initialization.
			net.test(di);
		}
	}

	const glades::NNetworkStatus st = net.saveModel(std::string(modelName.c_str()));
	if (!st.ok())
	{
		const shmea::GString msg = "Save failed: " + shmea::GString(st.message.c_str());
		RUMsgBox::MsgBox(this, "Model Package", msg, RUMsgBox::MESSAGEBOX);
	}
	else
	{
		const shmea::GString msg = "Saved \"" + modelName + "\"";
		RUMsgBox::MsgBox(this, "Model Package", msg, RUMsgBox::MESSAGEBOX);
	}

	if (di)
		delete di;

	loadDDNN();
}

/*!
 * @brief on Index to Edit change
 * @details runs when ddIndexToEdit changes value
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::clickedEditSwitch(const shmea::GString& cmpName, int x, int y)
{
	int indexToEdit = ddIndexToEdit->getSelectedIndex();
	if (indexToEdit == currentHiddenLayerIndex)
		return;

	syncFormVar();
	currentHiddenLayerIndex = indexToEdit;
	populateInputLayerForm();
	populateHLayerForm();
}

void NNCreatorPanel::clickedDSTypeSwitch(int newIndex)
{
	loadDatasets();
}

/*!
 * @brief event handler 4
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::clickedRun(const shmea::GString& cmpName, int x, int y)
{
	if (!serverInstance)
		return;

	if (!ddDataType)
		return;

	if (!ddDatasets)
		return;

	// Get the import type
	int importType = glades::DataInput::CSV;
	if (ddDataType->getSelectedText() == "CSV")
	{
		importType = glades::DataInput::CSV;
	}
	else if (ddDataType->getSelectedText() == "Image")
	{
		importType = glades::DataInput::IMAGE;
	}
	else if (ddDataType->getSelectedText() == "Text")
	{
		importType = glades::DataInput::TEXT;
	}
	else
		return;

	// Get the connection
	shmea::GString serverIP = "127.0.0.1";
	GNet::Connection* cConnection = serverInstance->getConnection(serverIP);
	if (!cConnection)
		return;

	// Get the vars from the components
	shmea::GString netName = tbNetName->getText();
	shmea::GString testFName = ddDatasets->getSelectedText();
	int64_t trainPct =
		parsePct(shmea::GString::Typify(tbTrainPct->getText(), tbTrainPct->getText().size()));
	int64_t testPct =
		parsePct(shmea::GString::Typify(tbTestPct->getText(), tbTestPct->getText().size()));
	int64_t validationPct = parsePct(
		shmea::GString::Typify(tbValidationPct->getText(), tbValidationPct->getText().size()));

	if ((netName.length() == 0) || (testFName.length() == 0))
		return;

	if (trainPct == -1 || testPct == -1 || validationPct == -1 ||
		trainPct + testPct + validationPct != 100)
	{
		printf("Percentages invalid: \n\tTrain: %s \n\tTest: %s \n\tValidation: "
			   "%s \n",
			   tbTrainPct->getText().c_str(), tbTestPct->getText().c_str(),
			   tbValidationPct->getText().c_str());
		return;
	}

	// Get the event listener ready
	resetSim();
	keepGraping = true;

	// Run a machine learning service
	shmea::GList wData;
	wData.addString(netName);
	wData.addString(testFName);
	wData.addInt(importType);

	// Modern config payload (kept simple and order-stable).
	// ML_Train will treat these as optional if absent.
	wData.addInt(ddNetType ? ddNetType->getSelectedIndex() : glades::NNetwork::TYPE_DFF);
	wData.addInt(ddLRSchedule ? scheduleTypeFromIndex(ddLRSchedule->getSelectedIndex()) : 0);
	wData.addInt(tbStepSize ? shmea::GString::Typify(tbStepSize->getText(), tbStepSize->getText().size()).getInt() : 0);
	wData.addFloat(tbGamma ? shmea::GString::Typify(tbGamma->getText(), tbGamma->getText().size()).getFloat() : 1.0f);
	wData.addInt(tbTMax ? shmea::GString::Typify(tbTMax->getText(), tbTMax->getText().size()).getInt() : 0);
	wData.addFloat(tbMinMult ? shmea::GString::Typify(tbMinMult->getText(), tbMinMult->getText().size()).getFloat() : 0.0f);
	wData.addFloat(tbGradClipNorm ? shmea::GString::Typify(tbGradClipNorm->getText(), tbGradClipNorm->getText().size()).getFloat() : 0.0f);
	wData.addFloat(tbPerElemClip ? shmea::GString::Typify(tbPerElemClip->getText(), tbPerElemClip->getText().size()).getFloat() : 10.0f);
	wData.addInt(tbTBPTT ? shmea::GString::Typify(tbTBPTT->getText(), tbTBPTT->getText().size()).getInt() : 0);
	// TrainingConfig extras (v2+)
	wData.addInt(tbMinibatchOverride ? shmea::GString::Typify(tbMinibatchOverride->getText(), tbMinibatchOverride->getText().size()).getInt() : 0);
	wData.addInt(ddOptimizer ? ddOptimizer->getSelectedIndex() : 0);
	wData.addFloat(tbAdamBeta1 ? shmea::GString::Typify(tbAdamBeta1->getText(), tbAdamBeta1->getText().size()).getFloat() : 0.9f);
	wData.addFloat(tbAdamBeta2 ? shmea::GString::Typify(tbAdamBeta2->getText(), tbAdamBeta2->getText().size()).getFloat() : 0.999f);
	wData.addFloat(tbAdamEps ? shmea::GString::Typify(tbAdamEps->getText(), tbAdamEps->getText().size()).getFloat() : 1e-8f);
	wData.addInt((chkAdamBiasCorrection && chkAdamBiasCorrection->isChecked()) ? 1 : 0);

	wData.addInt(tbTrHeads ? shmea::GString::Typify(tbTrHeads->getText(), tbTrHeads->getText().size()).getInt() : 0);
	wData.addInt(tbTrKVHeads ? shmea::GString::Typify(tbTrKVHeads->getText(), tbTrKVHeads->getText().size()).getInt() : 0);
	wData.addInt(tbTrDFF ? shmea::GString::Typify(tbTrDFF->getText(), tbTrDFF->getText().size()).getInt() : 0);
	wData.addInt((chkTrTokenEmbedding && chkTrTokenEmbedding->isChecked()) ? 1 : 0);
	wData.addInt(tbTrVocabSize ? shmea::GString::Typify(tbTrVocabSize->getText(), tbTrVocabSize->getText().size()).getInt() : 0);
	wData.addInt((chkTrTieEmbeddings && chkTrTieEmbeddings->isChecked()) ? 1 : 0);
	wData.addInt(tbTrPadTokenId ? shmea::GString::Typify(tbTrPadTokenId->getText(), tbTrPadTokenId->getText().size()).getInt() : -1);
	wData.addInt(ddTrPosEnc ? ddTrPosEnc->getSelectedIndex() : 1);
	wData.addInt(ddTrNorm ? ddTrNorm->getSelectedIndex() : 0);
	wData.addInt(ddTrFFNKind ? ddTrFFNKind->getSelectedIndex() : 0);
	wData.addInt(ddTrFFNAct ? ddTrFFNAct->getSelectedIndex() : 0);
	wData.addInt(ddTrKVCacheDType ? ddTrKVCacheDType->getSelectedIndex() : 0);
	wData.addInt(tbTrRoPEDim ? shmea::GString::Typify(tbTrRoPEDim->getText(), tbTrRoPEDim->getText().size()).getInt() : 0);
	wData.addFloat(tbTrRoPETheta ? shmea::GString::Typify(tbTrRoPETheta->getText(), tbTrRoPETheta->getText().size()).getFloat() : 10000.0f);
	wData.addInt(ddTrLossKind ? ddTrLossKind->getSelectedIndex() : 0);
	wData.addInt(tbTrNegSamples ? shmea::GString::Typify(tbTrNegSamples->getText(), tbTrNegSamples->getText().size()).getInt() : 64);

	/*wData.addLong(trainPct);
	wData.addLong(testPct);
	wData.addLong(validationPct);*/
	shmea::ServiceData* cSrvc = new shmea::ServiceData(cConnection, "ML_Train");
	cSrvc->set("net" + shmea::GString::intTOstring(netCount), wData);
	serverInstance->send(cSrvc);
	++netCount;
}

void NNCreatorPanel::clickedContinue(const shmea::GString& cmpName, int x, int y)
{
	if (!serverInstance)
		return;

	if (netCount == 0)
		return;

	if (!ddDataType)
		return;

	if (!ddDatasets)
		return;

	// Get the import type
	int importType = glades::DataInput::CSV;
	if (ddDataType->getSelectedText() == "CSV")
	{
		importType = glades::DataInput::CSV;
	}
	else if (ddDataType->getSelectedText() == "Image")
	{
		importType = glades::DataInput::IMAGE;
	}
	else if (ddDataType->getSelectedText() == "Text")
	{
		importType = glades::DataInput::TEXT;
	}
	else
		return;

	shmea::GString serverIP = "127.0.0.1";
	GNet::Connection* cConnection = serverInstance->getConnection(serverIP);
	if (!cConnection)
		return;

	// Get the vars from the components
	shmea::GString netName = tbNetName->getText();
	shmea::GString testFName = ddDatasets->getSelectedText();
	int64_t trainPct =
		parsePct(shmea::GString::Typify(tbTrainPct->getText(), tbTrainPct->getText().size()));
	int64_t testPct =
		parsePct(shmea::GString::Typify(tbTestPct->getText(), tbTestPct->getText().size()));
	int64_t validationPct = parsePct(
		shmea::GString::Typify(tbValidationPct->getText(), tbValidationPct->getText().size()));

	if ((netName.length() == 0) || (testFName.length() == 0))
		return;

	if (trainPct == -1 || testPct == -1 || validationPct == -1 ||
		trainPct + testPct + validationPct != 100)
	{
		printf("Percentages invalid: \n\tTrain: %s \n\tTest: %s \n\tValidation: "
			   "%s \n",
			   tbTrainPct->getText().c_str(), tbTestPct->getText().c_str(),
			   tbValidationPct->getText().c_str());
		return;
	}

	// Get the event listener ready
	keepGraping = true;

	// Run a machine learning service
	shmea::GList wData;
	wData.addString(netName);
	wData.addString(testFName);
	wData.addInt(importType);

	// Modern config payload (optional).
	wData.addInt(ddNetType ? ddNetType->getSelectedIndex() : glades::NNetwork::TYPE_DFF);
	wData.addInt(ddLRSchedule ? scheduleTypeFromIndex(ddLRSchedule->getSelectedIndex()) : 0);
	wData.addInt(tbStepSize ? shmea::GString::Typify(tbStepSize->getText(), tbStepSize->getText().size()).getInt() : 0);
	wData.addFloat(tbGamma ? shmea::GString::Typify(tbGamma->getText(), tbGamma->getText().size()).getFloat() : 1.0f);
	wData.addInt(tbTMax ? shmea::GString::Typify(tbTMax->getText(), tbTMax->getText().size()).getInt() : 0);
	wData.addFloat(tbMinMult ? shmea::GString::Typify(tbMinMult->getText(), tbMinMult->getText().size()).getFloat() : 0.0f);
	wData.addFloat(tbGradClipNorm ? shmea::GString::Typify(tbGradClipNorm->getText(), tbGradClipNorm->getText().size()).getFloat() : 0.0f);
	wData.addFloat(tbPerElemClip ? shmea::GString::Typify(tbPerElemClip->getText(), tbPerElemClip->getText().size()).getFloat() : 10.0f);
	wData.addInt(tbTBPTT ? shmea::GString::Typify(tbTBPTT->getText(), tbTBPTT->getText().size()).getInt() : 0);
	// TrainingConfig extras (v2+)
	wData.addInt(tbMinibatchOverride ? shmea::GString::Typify(tbMinibatchOverride->getText(), tbMinibatchOverride->getText().size()).getInt() : 0);
	wData.addInt(ddOptimizer ? ddOptimizer->getSelectedIndex() : 0);
	wData.addFloat(tbAdamBeta1 ? shmea::GString::Typify(tbAdamBeta1->getText(), tbAdamBeta1->getText().size()).getFloat() : 0.9f);
	wData.addFloat(tbAdamBeta2 ? shmea::GString::Typify(tbAdamBeta2->getText(), tbAdamBeta2->getText().size()).getFloat() : 0.999f);
	wData.addFloat(tbAdamEps ? shmea::GString::Typify(tbAdamEps->getText(), tbAdamEps->getText().size()).getFloat() : 1e-8f);
	wData.addInt((chkAdamBiasCorrection && chkAdamBiasCorrection->isChecked()) ? 1 : 0);

	wData.addInt(tbTrHeads ? shmea::GString::Typify(tbTrHeads->getText(), tbTrHeads->getText().size()).getInt() : 0);
	wData.addInt(tbTrKVHeads ? shmea::GString::Typify(tbTrKVHeads->getText(), tbTrKVHeads->getText().size()).getInt() : 0);
	wData.addInt(tbTrDFF ? shmea::GString::Typify(tbTrDFF->getText(), tbTrDFF->getText().size()).getInt() : 0);
	wData.addInt((chkTrTokenEmbedding && chkTrTokenEmbedding->isChecked()) ? 1 : 0);
	wData.addInt(tbTrVocabSize ? shmea::GString::Typify(tbTrVocabSize->getText(), tbTrVocabSize->getText().size()).getInt() : 0);
	wData.addInt((chkTrTieEmbeddings && chkTrTieEmbeddings->isChecked()) ? 1 : 0);
	wData.addInt(tbTrPadTokenId ? shmea::GString::Typify(tbTrPadTokenId->getText(), tbTrPadTokenId->getText().size()).getInt() : -1);
	wData.addInt(ddTrPosEnc ? ddTrPosEnc->getSelectedIndex() : 1);
	wData.addInt(ddTrNorm ? ddTrNorm->getSelectedIndex() : 0);
	wData.addInt(ddTrFFNKind ? ddTrFFNKind->getSelectedIndex() : 0);
	wData.addInt(ddTrFFNAct ? ddTrFFNAct->getSelectedIndex() : 0);
	wData.addInt(ddTrKVCacheDType ? ddTrKVCacheDType->getSelectedIndex() : 0);
	wData.addInt(tbTrRoPEDim ? shmea::GString::Typify(tbTrRoPEDim->getText(), tbTrRoPEDim->getText().size()).getInt() : 0);
	wData.addFloat(tbTrRoPETheta ? shmea::GString::Typify(tbTrRoPETheta->getText(), tbTrRoPETheta->getText().size()).getFloat() : 10000.0f);
	wData.addInt(ddTrLossKind ? ddTrLossKind->getSelectedIndex() : 0);
	wData.addInt(tbTrNegSamples ? shmea::GString::Typify(tbTrNegSamples->getText(), tbTrNegSamples->getText().size()).getInt() : 64);
	/*wData.addLong(trainPct);
	wData.addLong(testPct);
	wData.addLong(validationPct);*/
	shmea::ServiceData* cSrvc = new shmea::ServiceData(cConnection, "ML_Train");
	cSrvc->set("net" + shmea::GString::intTOstring(netCount - 1), wData);
	serverInstance->send(cSrvc);
}

/*!
 * @brief event handler 6
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::clickedCopy(const shmea::GString& cmpName, int x, int y)
{
	syncFormVar();
	shmea::GType destination(tbCopyDestination->getText());
	if (destination.getType() != shmea::GType::LONG_TYPE)
		return;

	int dst = destination.getInt();
	if (dst < 0 || dst >= formInfo->numHiddenLayers() || dst == currentHiddenLayerIndex)
		return;

	formInfo->copyHiddenLayer(dst, currentHiddenLayerIndex);
	printf("[GUI] Layer %d parameters copied to layer %d\n", currentHiddenLayerIndex, dst);
}

/*!
 * @brief event handler 7
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::clickedRemove(const shmea::GString& cmpName, int x, int y)
{
	if (formInfo->numHiddenLayers() <= 1)
		return;

	formInfo->removeHiddenLayer(currentHiddenLayerIndex);
	if (currentHiddenLayerIndex > formInfo->numHiddenLayers())
		--currentHiddenLayerIndex;

	printf("[GUI] Layer %d deleted\n", currentHiddenLayerIndex);
	populateInputLayerForm();
	populateHLayerForm();
	tbHiddenLayerCount->setText(shmea::GString::intTOstring(formInfo->numHiddenLayers()));
	populateIndexToEdit(currentHiddenLayerIndex);
}

/*!
 * @brief Hidden Layer Count change handler
 * @details runs whenever the tbHiddenLayerCount changes (i.e. it loses focus).
 * For now, changes
 * hiddenLayers, currentHiddenLayerIndex, and formInfo->numHiddenLayers()
 * accordingly
 */
void NNCreatorPanel::tbHLLoseFocus()
{
	shmea::GType newHiddenLayerCount =
		shmea::GString::Typify(tbHiddenLayerCount->getText(), tbHiddenLayerCount->getText().size());
	if (newHiddenLayerCount.getType() != shmea::GType::LONG_TYPE)
		return;

	int newCount = (int)newHiddenLayerCount.getLong();
	if (newCount < 0 || (newCount == formInfo->numHiddenLayers()))
		return;

	formInfo->resizeHiddenLayers(newCount);
	if (currentHiddenLayerIndex >= newCount)
	{
		currentHiddenLayerIndex = formInfo->numHiddenLayers() - 1;
		populateInputLayerForm();
		populateHLayerForm();
	}

	populateIndexToEdit(currentHiddenLayerIndex);
}

void NNCreatorPanel::clickedLoad(const shmea::GString& cmpName, int x, int y)
{
	loadDDNN();
}

void NNCreatorPanel::checkedCV(const shmea::GString& cmpName, int x, int y)
{
	// Hide cross validate options if visible
	if (lblttv->isVisible())
	{
		lblttv->setVisible(false);
		tbTrainPct->setVisible(false);
		tbTestPct->setVisible(false);
		tbValidationPct->setVisible(false);
	}
	else // Show cross validate options if hidden
	{
		lblttv->setVisible(true);
		tbTrainPct->setVisible(true);
		tbTestPct->setVisible(true);
		tbValidationPct->setVisible(true);
	}
}

void NNCreatorPanel::clickedKill(const shmea::GString& cmpName, int x, int y)
{
	shmea::GString serverIP = "127.0.0.1";
	GNet::Connection* cConnection = serverInstance->getConnection(serverIP);
	if (!cConnection)
		return;

	keepGraping = false;

	// Kill a neural network instance
	shmea::GList wData;
	wData.addString("KILL");

	shmea::ServiceData* cSrvc = new shmea::ServiceData(cConnection, "ML_Train");
	cSrvc->set("net" + shmea::GString::intTOstring(netCount - 1), wData);
	serverInstance->send(cSrvc);
}

void NNCreatorPanel::clickedDelete(const shmea::GString& cmpName, int x, int y)
{
	shmea::GString netName = tbNetName->getText();
	const std::string path = std::string("database/models/") + std::string(netName.c_str());
	const bool ok = deleteRecursive(path);
	loadDDNN();

	// Display a popup alert
	if (ok)
	{
		const shmea::GString msgBoxText = "Deleted \"" + netName + "\"";
		RUMsgBox::MsgBox(this, "Model Package", msgBoxText, RUMsgBox::MESSAGEBOX);
	}
	else
	{
		const shmea::GString msgBoxText = "Delete failed \"" + netName + "\"";
		RUMsgBox::MsgBox(this, "Model Package", msgBoxText, RUMsgBox::MESSAGEBOX);
	}
}

void NNCreatorPanel::clickedPreviewTrain(const shmea::GString& cmpName, int x, int y)
{
	if (!ddDatasets)
		return;

	shmea::GString testFName = ddDatasets->getSelectedText();

	if (testFName.length() == 0)
		return;

	ii.import(testFName.c_str());
	prevImageFlag = 0;
	previewImage->setBGImage(ii.getTrainImage(trainingRowIndex));
}

void NNCreatorPanel::clickedPreviewTest(const shmea::GString& cmpName, int x, int y)
{
	if (!ddDatasets)
		return;

	shmea::GString testFName = ddDatasets->getSelectedText();

	if (testFName.length() == 0)
		return;

	ii.import(testFName.c_str());
	prevImageFlag = 1;
	previewImage->setBGImage(ii.getTestImage(testingRowIndex));
}

void NNCreatorPanel::clickedPrevious(const shmea::GString& cmpName, int x, int y)
{
	if (prevImageFlag == 0)
	{
		if (trainingRowIndex > 0)
			--trainingRowIndex;
		previewImage->setBGImage(ii.getTrainImage(trainingRowIndex));
	}
	else if (prevImageFlag == 1)
	{
		if (testingRowIndex > 0)
			--testingRowIndex;
		previewImage->setBGImage(ii.getTestImage(testingRowIndex));
	}
}

void NNCreatorPanel::clickedNext(const shmea::GString& cmpName, int x, int y)
{
	if (prevImageFlag == 0)
	{
		if (trainingRowIndex < ii.getTrainSize() - 1)
			++trainingRowIndex;
		previewImage->setBGImage(ii.getTrainImage(trainingRowIndex));
	}
	else if (prevImageFlag == 1)
	{
		if (testingRowIndex < ii.getTestSize() - 1)
			++testingRowIndex;
		previewImage->setBGImage(ii.getTestImage(testingRowIndex));
	}
}

void NNCreatorPanel::nnSelectorChanged(int newIndex)
{
	// Only load after a selection click (not an open click)
	if (ddNeuralNet->isOpen())
		return;

	// make sure the user wants to actually load a network, since index 0 = "New"
	int loadOrSave = ddNeuralNet->getSelectedIndex();
	if (loadOrSave == 0)
		return;

	shmea::GString netName = ddNeuralNet->getSelectedText();
	tbNetName->setText(netName);
	formInfo->setName(netName.c_str());

	// Load NNInfo from the unified model package.
	const std::string modelName(netName.c_str());
	const std::string base = std::string("database/models/") + modelName + "/";
	const std::string nninfoPath = base + "nninfo.csv";

	// Attempt to load the full model (weights + TrainingConfig) via the production API.
	// This requires a DataInput instance for shaping tensors.
	glades::DataInput* di = NULL;
	if (ddDataType && ddDatasets && ddDatasets->getSelectedText().length() > 0)
	{
		shmea::GString inputFName = ddDatasets->getSelectedText();
		const shmea::GString dtype = ddDataType->getSelectedText();
		if (dtype == "CSV")
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::NumberInput();
		}
		else if (dtype == "Image")
		{
			di = new glades::ImageInput();
		}
		else if (dtype == "Text")
		{
			inputFName = "datasets/" + inputFName;
			glades::TokenInput* ti = new glades::TokenInput();
			// Use current UI pad token id for shaping next-token targets (best-effort).
			const int padId = (tbTrPadTokenId ? shmea::GString::Typify(tbTrPadTokenId->getText(), tbTrPadTokenId->getText().size()).getInt() : -1);
			ti->setPadTokenId(padId);
			di = ti;
		}

		if (di)
			di->import(inputFName);
	}

	bool loadedFull = false;
	if (di)
	{
		glades::NNetwork net(glades::NNetwork::TYPE_DFF);
		const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
		if (!stLoad.ok())
		{
			const shmea::GString msg = "Load failed: " + shmea::GString(stLoad.message.c_str());
			RUMsgBox::MsgBox(this, "Model Package", msg, RUMsgBox::MESSAGEBOX);
		}
		else
		{
			loadedFull = true;

			if (ddNetType)
				ddNetType->setSelectedIndex(net.getNetType());

			const glades::TrainingConfig& cfg = net.getTrainingConfig();
			if (tbMinibatchOverride)
				tbMinibatchOverride->setText(shmea::GString::intTOstring(cfg.minibatchSizeOverride));
			if (tbTBPTT)
				tbTBPTT->setText(shmea::GString::intTOstring(cfg.tbpttWindowOverride));
			if (tbGradClipNorm)
				tbGradClipNorm->setText(shmea::GString::floatTOstring(cfg.globalGradClipNorm));
			if (tbPerElemClip)
				tbPerElemClip->setText(shmea::GString::floatTOstring(cfg.perElementGradClip));

			if (ddLRSchedule)
				ddLRSchedule->setSelectedIndex(static_cast<int>(cfg.lrSchedule.type));
			if (tbStepSize)
				tbStepSize->setText(shmea::GString::intTOstring(cfg.lrSchedule.stepSizeEpochs));
			if (tbGamma)
				tbGamma->setText(shmea::GString::floatTOstring(cfg.lrSchedule.gamma));
			if (tbTMax)
				tbTMax->setText(shmea::GString::intTOstring(cfg.lrSchedule.cosineTMaxEpochs));
			if (tbMinMult)
				tbMinMult->setText(shmea::GString::floatTOstring(cfg.lrSchedule.minMultiplier));

			// Optimizer
			if (ddOptimizer)
				ddOptimizer->setSelectedIndex(static_cast<int>(cfg.optimizer.type));
			if (tbAdamBeta1)
				tbAdamBeta1->setText(shmea::GString::floatTOstring(cfg.optimizer.adamBeta1));
			if (tbAdamBeta2)
				tbAdamBeta2->setText(shmea::GString::floatTOstring(cfg.optimizer.adamBeta2));
			if (tbAdamEps)
				tbAdamEps->setText(shmea::GString::floatTOstring(cfg.optimizer.adamEps));
			if (chkAdamBiasCorrection)
				chkAdamBiasCorrection->setCheck(cfg.optimizer.adamBiasCorrection);

			// Transformer
			if (tbTrHeads)
				tbTrHeads->setText(shmea::GString::intTOstring(cfg.transformer.nHeadsOverride));
			if (tbTrKVHeads)
				tbTrKVHeads->setText(shmea::GString::intTOstring(cfg.transformer.nKVHeadsOverride));
			if (tbTrDFF)
				tbTrDFF->setText(shmea::GString::intTOstring(cfg.transformer.dFFOverride));

			if (chkTrTokenEmbedding)
				chkTrTokenEmbedding->setCheck(cfg.transformer.enableTokenEmbedding);
			if (tbTrVocabSize)
				tbTrVocabSize->setText(shmea::GString::intTOstring(cfg.transformer.vocabSizeOverride));
			if (chkTrTieEmbeddings)
				chkTrTieEmbeddings->setCheck(cfg.transformer.tieEmbeddings);
			if (tbTrPadTokenId)
				tbTrPadTokenId->setText(shmea::GString::intTOstring(cfg.transformer.padTokenId));

			if (ddTrPosEnc)
				ddTrPosEnc->setSelectedIndex(static_cast<int>(cfg.transformer.positionalEncoding));
			if (ddTrNorm)
				ddTrNorm->setSelectedIndex(static_cast<int>(cfg.transformer.normType));
			if (ddTrFFNKind)
				ddTrFFNKind->setSelectedIndex(static_cast<int>(cfg.transformer.ffnKind));
			if (ddTrFFNAct)
				ddTrFFNAct->setSelectedIndex(static_cast<int>(cfg.transformer.ffnActivation));
			if (ddTrKVCacheDType)
				ddTrKVCacheDType->setSelectedIndex(static_cast<int>(cfg.transformer.kvCacheDType));
			if (tbTrRoPEDim)
				tbTrRoPEDim->setText(shmea::GString::intTOstring(cfg.transformer.ropeDimOverride));
			if (tbTrRoPETheta)
				tbTrRoPETheta->setText(shmea::GString::floatTOstring(cfg.transformer.ropeTheta));
			if (ddTrLossKind)
				ddTrLossKind->setSelectedIndex(static_cast<int>(cfg.transformer.tokenLmLossKind));
			if (tbTrNegSamples)
				tbTrNegSamples->setText(shmea::GString::intTOstring(cfg.transformer.tokenLmSampledNegatives));
		}
	}
	else
	{
		RUMsgBox::MsgBox(this, "Model Package",
		                 "Select a dataset to fully load weights/config (loadModel requires DataInput).",
		                 RUMsgBox::MESSAGEBOX);
	}

	if (di)
		delete di;

	// Always load the NNInfo table for editing/preview.
	shmea::GTable tab(nninfoPath.c_str(), ',', shmea::GTable::TYPE_FILE);
	glades::NNInfo* info = new glades::NNInfo(netName, tab);
	loadNNet(info);

	// If loadNNet() overwrote the name box, keep it aligned.
	if (loadedFull && tbNetName)
		tbNetName->setText(netName);
}

/*!
 * @brief Plot the Learning Curve of the NN
 * @details Plot the learning curve of the neural network on the graph
 */
void NNCreatorPanel::PlotLearningCurve(float newXVal, float newYVal)
{
	if (!lcGraph)
		return;

	pthread_mutex_lock(lcMutex);

	SDL_Color lineColor = RUColors::DEFAULT_COLOR_LINE;
	lcGraph->add("lc", new Point2(newXVal, newYVal), lineColor);

	pthread_mutex_unlock(lcMutex);
}

/*!
 * @brief Plot the ROC Curve of the NN
 * @details Plot the ROC Curve of the neural network on the graph
 */
void NNCreatorPanel::PlotROCCurve(float newXVal, float newYVal)
{
	if (!rocCurveGraph)
		return;

	pthread_mutex_lock(rocMutex);

	SDL_Color lineColor = RUColors::DEFAULT_COLOR_LINE;
	rocCurveGraph->add("roc", new Point2(newXVal, newYVal), lineColor);

	pthread_mutex_unlock(rocMutex);
}

void NNCreatorPanel::updateConfMatrixTable(const shmea::GTable& newMatrix)
{
	cMatrixTable->import(newMatrix);
	cMatrixTable->updateLabels();
}

void NNCreatorPanel::updateFromQ(const shmea::ServiceData* data)
{
	shmea::GList argList = data->getArgList();
	if (argList.size() == 0)
		return;

	shmea::GString cName = argList.getString(0);

	if (cName == "RESET")
	{
		// Special case to reset the sim GUI
		resetSim();
	}
	else if (cName == "UPDATE-GRAPHS")
	{
		if (!keepGraping)
			return;

		// Special case to update the candle graph
		lcGraph->update();
		rocCurveGraph->update();
	}
	else if (cName == "ACC")
	{
		if (!keepGraping)
			return;

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return;

		// Update components by tick
		shmea::GList cList = data->getList();
		if (cList.size() < 2)
			return;

		int epochs = cList.getInt(0);
		float accuracy = cList.getFloat(1);
		char accBuf[64];
		sprintf(accBuf, "%.2f", accuracy);
		lblEpochs->setText(shmea::GString::intTOstring(epochs) + "(t)");
		lblAccuracy->setText(shmea::GString(accBuf) + "% Accuracy");
	}
	else if (cName == "CONF")
	{
		if (!keepGraping)
			return;

		if (data->getType() != shmea::ServiceData::TYPE_TABLE)
			return;

		shmea::GList argList = data->getArgList();
		if (argList.size() < 2)
			return;

		// Closing Price Label
		float falseAlarm = argList.getFloat(0);
		float recall = argList.getFloat(1);

		PlotROCCurve(falseAlarm, recall);
		updateConfMatrixTable(data->getTable());
	}
	else if (cName == "PROGRESSIVE")
	{
		if (!keepGraping)
			return;

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return;

		// Update components by tick
		shmea::GList cList = data->getList();
		if (cList.size() < 2)
			return;

		// Closing Price Label
		int newXVal = cList.getInt(0);
		float lcPoint = cList.getFloat(1);

		// Graphs
		PlotLearningCurve(newXVal, lcPoint);
	}
	else if (cName == "ACTIVATIONS")
	{

		shmea::GList activations = data->getList();
		if (activations[0].getType() == shmea::GType::INT_TYPE)
		{
			for (unsigned int i = 0; i < activations.size(); i++)
			{
				// Initialize the neural network visualizer
				if (activations[i].getType() == shmea::GType::INT_TYPE)
				{
					if (i == 0)
					{
						nn = new DrawNeuralNet(activations.size());
						nn->setInputLayer(activations.getInt(i));
					}
					else if (i == activations.size() - 1)
					{
						nn->setOutputLayer(activations.getInt(i));
					}
					else
					{
						nn->setHiddenLayer(i, activations.getInt(i));
					}
				}
			}
		}
		else
		{
			nn->setActivation(activations);
			neuralNetGraph->set("nn", nn);
		}

		// nn->displayNeuralNet(); // DEBUGGING ONLY
	}
	else if (cName == "WEIGHTS")
	{

		// Update weights
		shmea::GList weights = data->getList();

		if (weights.size() < 1 && nn == NULL)
			return;

		nn->setWeights(weights);
		// nn->displayNeuralNet(); // DEBUGGING ONLY
		neuralNetGraph->set("nn", nn);
	}
}

void NNCreatorPanel::resetSim()
{
	pthread_mutex_lock(lcMutex);
	lcGraph->clear();
	lcGraph->update();
	pthread_mutex_unlock(lcMutex);

	pthread_mutex_lock(rocMutex);
	rocCurveGraph->clear();
	rocCurveGraph->update();
	pthread_mutex_unlock(rocMutex);

	// Reset the NN updates
	pthread_mutex_lock(qMutex);
	std::queue<const shmea::ServiceData*> emptyQ;
	std::swap(updateQueue, emptyQ);
	pthread_mutex_unlock(qMutex);

	lblEpochs->setText("0(t)");
	lblAccuracy->setText("N/A Accuracy");
}
