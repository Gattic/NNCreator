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
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "Backend/Machine Learning/DataObjects/ImageInput.h"
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

// using namespace shmea;
using namespace glades;

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

	GLinearLayout* loadNetLayout = new GLinearLayout("loadNetLayout");
	loadNetLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(loadNetLayout);

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
	leftSideLayout->addSubItem(netNameLayout);
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

	GLinearLayout* loadNetLayoutWeights = new GLinearLayout("loadNetLayoutWeights");
	loadNetLayoutWeights->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(loadNetLayoutWeights);

	// Neural Net selector label
	lblNeuralNetWeights = new RULabel();
	lblNeuralNetWeights->setWidth(200);
	lblNeuralNetWeights->setHeight(30);
	lblNeuralNetWeights->setText("Load NN Weights");
	lblNeuralNetWeights->setName("lblNeuralNetWeights");
	loadNetLayoutWeights->addSubItem(lblNeuralNetWeights);

	// Neural Net selector
	ddNeuralNetWeights = new RUDropdown();
	ddNeuralNetWeights->setWidth(220);
	ddNeuralNetWeights->setHeight(30);
	ddNeuralNetWeights->setOptionsShown(3);
	ddNeuralNetWeights->setName("ddNeuralNetWeights");
	ddNeuralNetWeights->setOptionChangedListener(
		GeneralListener(this, &NNCreatorPanel::nnWeightsSelectorChanged));
	loadNetLayoutWeights->addSubItem(ddNeuralNetWeights);

	// Load button
	RUButton* btnLoadWeights = new RUButton();
	btnLoadWeights->setWidth(206);
	btnLoadWeights->setHeight(30);
	btnLoadWeights->setText("      Reload List");
	btnLoadWeights->setMouseDownListener(GeneralListener(this, &NNCreatorPanel::clickedLoad));
	btnLoadWeights->setName("btnLoad");
	loadNetLayoutWeights->addSubItem(btnLoadWeights);

	GLinearLayout* netNameLayoutWeights = new GLinearLayout("netNameLayout");
	netNameLayoutWeights->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(netNameLayoutWeights);
	// Network Name label
	lblNetNameWeights = new RULabel();
	lblNetNameWeights->setX(6);
	lblNetNameWeights->setY(6 + lblNeuralNet->getY() + lblNeuralNet->getHeight());
	lblNetNameWeights->setWidth(200);
	lblNetNameWeights->setHeight(30);
	lblNetNameWeights->setText("NN Weights");
	lblNetNameWeights->setName("lblNetNameWeights");
	netNameLayoutWeights->addSubItem(lblNetNameWeights);

	// Network Name textbox
	tbNetNameWeights = new RUTextbox();
	tbNetNameWeights->setX(20 + lblNetNameWeights->getX() + lblNetNameWeights->getWidth());
	tbNetNameWeights->setY(lblNetName->getY());
	tbNetNameWeights->setWidth(220);
	tbNetNameWeights->setHeight(30);
	tbNetNameWeights->setText("");
	tbNetNameWeights->setName("tbNetNameWeights");
	netNameLayoutWeights->addSubItem(tbNetNameWeights);

	// Save button
	btnSaveWeights = new RUButton("green");
	btnSaveWeights->setWidth(100);
	btnSaveWeights->setHeight(30);
	btnSaveWeights->setText("    Save");
	btnSaveWeights->setMouseDownListener(
		GeneralListener(this, &NNCreatorPanel::clickedSaveWeights));
	btnSaveWeights->setName("btnSaveWeights");
	netNameLayoutWeights->addSubItem(btnSaveWeights);

	// Delete button
	btnDeleteWeights = new RUButton("red");
	btnDeleteWeights->setWidth(100);
	btnDeleteWeights->setHeight(30);
	btnDeleteWeights->setText("  Delete");
	btnDeleteWeights->setMouseDownListener(
		GeneralListener(this, &NNCreatorPanel::clickedDeleteWeights));
	btnDeleteWeights->setName("btnDeleteWeights");
	netNameLayoutWeights->addSubItem(btnDeleteWeights);

	// Input/Output Data Type Layout
	GLinearLayout* dataTypeLayout = new GLinearLayout("dataTypeLayout");
	dataTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(dataTypeLayout);

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
	leftSideLayout->addSubItem(runTestLayout);

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
	// leftSideLayout->addSubItem(crossValLayout); // TODO uncomment when cross val is tested

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

	// add items from nnetworks table to dropdown
	shmea::SaveFolder* nnList = new shmea::SaveFolder("neuralnetworks");
	nnList->load();
	std::vector<shmea::SaveTable*> saveTables = nnList->getItems();
	for (unsigned int i = 0; i < saveTables.size(); ++i)
	{
		shmea::SaveTable* cItem = saveTables[i];
		if (!cItem)
			continue;

		ddNeuralNet->addOption(cItem->getName());
	}

	// clear the old items
	ddNeuralNetWeights->clearOptions();

	// add the "new" item option
	ddNeuralNetWeights->addOption(" ");

	// add items from nnetworks table to dropdown
	shmea::SaveFolder* nnListWeights = new shmea::SaveFolder("nn-state");
	nnListWeights->load();
	std::vector<shmea::SaveTable*> saveTablesWeights = nnListWeights->getItems();
	for (unsigned int i = 0; i < saveTablesWeights.size(); ++i)
	{
		shmea::SaveTable* cItem = saveTablesWeights[i];
		if (!cItem)
			continue;

		ddNeuralNetWeights->addOption(cItem->getName());
	}
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

	shmea::GString serverIP = "127.0.0.1";

	// make sure the layers are up to date
	syncFormVar();

	shmea::GString netName = tbNetName->getText();
	//	formInfo->setName(netName.c_str());

	shmea::GType batchSize =
		shmea::GString::Typify(tbBatchSize->getText(), tbBatchSize->getText().size());
	if (batchSize.getType() != shmea::GType::LONG_TYPE)
		return;

	shmea::GType inputLR =
		shmea::GString::Typify(tbinputLR->getText(), tbinputLR->getText().size());
	if (inputLR.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType inputMF =
		shmea::GString::Typify(tbinputMF->getText(), tbinputMF->getText().size());
	if (inputMF.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType inputWD1 =
		shmea::GString::Typify(tbinputWD1->getText(), tbinputWD1->getText().size());
	if (inputWD1.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType inputWD2 =
		shmea::GString::Typify(tbinputWD2->getText(), tbinputWD2->getText().size());
	if (inputWD2.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType inputDropout =
		shmea::GString::Typify(tbinputDropout->getText(), tbinputDropout->getText().size());
	if (inputDropout.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType inputAP =
		shmea::GString::Typify(tbinputAP->getText(), tbinputAP->getText().size());
	if (inputAP.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType outputSize =
		shmea::GString::Typify(tbOutputLayerSize->getText(), tbOutputLayerSize->getText().size());
	if (outputSize.getType() != shmea::GType::LONG_TYPE)
		return;

	// just make sure the expected hidden layer counts are equal
	shmea::GType layerCountCheck =
		shmea::GString::Typify(tbHiddenLayerCount->getText(), tbHiddenLayerCount->getText().size());
	if (layerCountCheck.getType() != shmea::GType::LONG_TYPE)
		return;

	int newCount = (int)layerCountCheck.getLong();
	if (newCount != formInfo->numHiddenLayers())
		return;

	// Save the neural network
	populateInputLayerForm();
	populateHLayerForm();
	NNetwork* network = new NNetwork(formInfo);
	glades::saveNeuralNetwork(network);
	loadDDNN();
}

/*!
 * @brief Save NN weights to file
 * @details save a neural network bias and edge weights
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::clickedSaveWeights(const shmea::GString& cmpName, int x, int y)
{

	if (!tbNetNameWeights)
		return;

	if (tbNetNameWeights->getText().length() == 0)
		return;

	shmea::SaveFolder* folderToSave = new shmea::SaveFolder("nn-state");
	if (!folderToSave->checkFolder())
	{
		return;
	}

	std::ofstream out((folderToSave->getPath() + tbNetNameWeights->getText()).c_str());
	if (!out)
	{
		return;
	}

	shmea::GString serverIP = "127.0.0.1";

	// make sure the layers are up to date
	syncFormVar();

	shmea::GString netName = tbNetNameWeights->getText();

	int layerCount = nn->getLayersCount();
	for (int i = 1; i < layerCount; ++i)
	{
		float biasWeight = nn->getLayerBiasWeight(i);
		int lNeuronCount = nn->getLayerNeuronsCount(i);
		int neuronEdgeCount = 0;
		std::vector<shmea::GPointer<DrawNeuron> > neurons = nn->getLayerNeurons(i);
		if (lNeuronCount > 0 && neurons.size() > 0)
		{
			neuronEdgeCount = neurons[0]->getWeights().size();
		}

		out << biasWeight << " ";
		out << lNeuronCount << " ";
		out << neuronEdgeCount << "\n";

		for (int n = 0; n < lNeuronCount; ++n)
		{
			DrawNeuron neuron = *neurons[n];
			std::vector<shmea::GPointer<float> > cWeights = neuron.getWeights();
			for (int e = 0; e < cWeights.size(); ++e)
			{
				float w = *cWeights[e];
				out << w << " ";
			}
		}
		out << "\n";
	}
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
	shmea::GString netNameWeights = ddNeuralNetWeights->getSelectedText();
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
	if (netNameWeights.c_str() != " ")
		wData.addString(netNameWeights);

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
	shmea::GString netNameWeights = ddNeuralNetWeights->getSelectedText();
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
	if (netNameWeights.c_str() != " ")
		wData.addString(netNameWeights);
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
	shmea::SaveFolder* nnList = new shmea::SaveFolder("neuralnetworks");
	nnList->deleteItem(netName);
	loadDDNN();

	// Display a popup alert
	char buffer[netName.length()];
	sprintf(buffer, "Deleted \"%s\"", netName.c_str());
	RUMsgBox::MsgBox(this, "Neural Net", buffer, RUMsgBox::MESSAGEBOX);
}

void NNCreatorPanel::clickedDeleteWeights(const shmea::GString& cmpName, int x, int y)
{
	shmea::GString netName = tbNetNameWeights->getText();
	shmea::SaveFolder* nnList = new shmea::SaveFolder("nn-state");
	nnList->deleteItem(netName);
	loadDDNN();

	// Display a popup alert
	char buffer[netName.length()];
	sprintf(buffer, "Deleted \"%s\"", netName.c_str());
	RUMsgBox::MsgBox(this, "Neural Net", buffer, RUMsgBox::MESSAGEBOX);
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

	// Load a machine learning model
	glades::NNetwork cNetwork;
	if (!cNetwork.load(netName.c_str()))
	{
		printf("[NN] Unable to load \"%s\"", netName.c_str());
		return;
	}

	// Load the NN into the form
	loadNNet(cNetwork.getNNInfo());
}

void NNCreatorPanel::nnWeightsSelectorChanged(int newIndex)
{
	// Only load after a selection click (not an open click)
	if (ddNeuralNetWeights->isOpen())
		return;

	// make sure the user wants to actually load a network, since index 0 = " "
	int loadOrSave = ddNeuralNetWeights->getSelectedIndex();
	if (loadOrSave == 0)
		return;

	shmea::GString netName = ddNeuralNetWeights->getSelectedText();
	tbNetNameWeights->setText(netName);
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
