/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */

#include "CNeuralNet.h"

//included for RandomClamped()
#include "utils.h"
#include <random> 

/*******************************
** --> Neuron Constructor <-- **
*******************************/
Neuron::Neuron(int numberInputs) : numInputs(numberInputs), n_error(0)
{
	for (int i = 0; i < numberInputs ; ++i) // the +1 is for the additional weight used as a bias
	{
		weightVector.push_back(RandomClamped()); 

		//set all the previous dealta weights to 0 when neuron is initialized
		previousDeltas.push_back(0);
	}
}

/**************************************
** --> Neuron Layer Constructor <-- **
**************************************/
NeuronLayer::NeuronLayer(int numberNeurons, int numInputPerNeuron) : numNeurons(numberNeurons)
{
	for (int i = 0; i < numberNeurons; ++i)
	{
		neuronVector.push_back(Neuron(numInputPerNeuron));
	}
}

/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff) :
	_inputLayerSize(inputLayerSize), _hiddenLayerSize(hiddenLayerSize), _outputLayerSize(outputLayerSize), _lRate(lRate), _mse_cutoff(mse_cutoff)
	//you probably want to use an initializer list here
{
	/*********************************************
	** --> create the layers of the network <-- **
	**********************************************/
	
	// create hidden layer
	layersVector.push_back(NeuronLayer(_hiddenLayerSize, _inputLayerSize));

	// create output layer
	layersVector.push_back(NeuronLayer(_outputLayerSize, _hiddenLayerSize));

	// initialized the weights of the neurons in the two layers above
	initWeights(); 
	
}
/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() {
	//TODO
}
/**
 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights()
{
	// For each layer
	for (int i = 0; i < 2; ++i)
	{
		// For each neuron   
		for (int n = 0; n < layersVector[i].numNeurons; ++n)
		{
			// For each weight   
			for (int w = 0; w < layersVector[i].neuronVector[n].numInputs; ++w)
			{
				layersVector[i].neuronVector[n].weightVector[w] = RandomClamped(); //RandomClamped returns a random float in the range - 1 < n < 1
			}
		}
	}
}
/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/
void CNeuralNet::feedForward(std::vector<double> inputs) 
{
	vector<double> tempIO = inputs;

	// For each layer --> 0 = hidden; 1 = output
	for (int l = 0; l < 2; ++l)
	{
		//The output of the previous iteration is the input for the current
		inputs = tempIO;
		tempIO.clear();
		//Clear the output vector//
		//because we only want the output of the last layer to be stored in the outputVector
		outputsVector.clear();

		//For each neuron in the neuron layer...
		for (int n = 0; n < layersVector[l].numNeurons; ++n)
		{
			double sumInputs = 0.0;

			//For each weight in the neuron...
			for (int w = 0; w < layersVector[l].neuronVector[n].numInputs; ++w)
			{
				//       sum of the w-th input with its respective weight w
				sumInputs += inputs[w] * layersVector[l].neuronVector[n].weightVector[w];
			}

			//Store the output of the neuron in the neuron
			layersVector[l].neuronVector[n].n_output = (1 / (1 + exp(-sumInputs)));			//sigmoid function --> (1 / (1 + exp(-sumInputs)))
			//Push the output of each neuron in the layer to the outputsVector 
			outputsVector.push_back(layersVector[l].neuronVector[n].n_output);
			//set temp IO for loop
			tempIO.push_back(layersVector[l].neuronVector[n].n_output);
		}
	}
}
/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * error at the hidden layer
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/
void CNeuralNet::propagateErrorBackward(std::vector<double> desiredOutput, std::vector<double> inputsLayer)
{
	// 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs) for each output.//

	//For each neuron in the output layer...
	for (int n = 0; n < _outputLayerSize; ++n)
	{
		//Calculate the error using sigmoid_d(output) * (difference between expected and computed outputs)
		//i.e. Gradient descent method --> from notes: err = oi(1 - oi)(ti - oi)
		double error = outputsVector[n] * (1 - outputsVector[n]) * (desiredOutput[n] - outputsVector[n]);

		//Store the error calculated for the neuron in the neuron
		layersVector[1].neuronVector[n].n_error = error;
	}	

	//2. Compute the error at the hidden layer : sigmoid_d(hidden) * sum(weights_o_h * difference between expected output and computed output at output layer)
		//for each hidden layer node

	//For each neuron in the hidden layer...
	for (int n = 0; n < _hiddenLayerSize; ++n)
	{
		double error = 0;

		//ERR = oh(1 - oh) * sum(whi * erri)

		//Calculate the sum of the errors*weights in the output layer... 
		for (int i = 0; i < _outputLayerSize; ++i)
		{
			error += layersVector[1].neuronVector[i].n_error * layersVector[1].neuronVector[i].weightVector[n];
		}
		// Calculate the error by multiplying the sum(whi * erri) to the oh(1 - oh)
		error *= layersVector[0].neuronVector[n].n_error * (1 - layersVector[0].neuronVector[n].n_error);
		//Store the updated error in the neuron
		layersVector[0].neuronVector[n].n_error = error;
	}

	//3. Adjust the weights from the hidden to the output layer : learning rate * error at the output layer * error at the hidden layer
	//for each connection between the hidden and output layers
	double alpha = 0.9; //alpha value needs to be between 0-1
	//For each neuron in the layer...
	for (int n = 0; n < layersVector[1].neuronVector.size(); ++n)
	{

		//For each weight in the neuron...
		for (int w = 0; w < layersVector[1].neuronVector[n].weightVector.size(); ++w)
		{
			//double alpha = ((double)rand() / (RAND_MAX)); //alpha value needs to be between 0-1
			double momentum = alpha * layersVector[1].neuronVector[n].previousDeltas[w]; // calc momentum with alpha and previous weight
			double deltaWeight = (_lRate * layersVector[1].neuronVector[n].n_error * layersVector[0].neuronVector[w].n_output) + momentum;
			layersVector[1].neuronVector[n].previousDeltas[w] = deltaWeight; //store the delta weight for next back propagation

			layersVector[1].neuronVector[n].weightVector[w] += deltaWeight;
		}
	}
	
	//4. Adjust the weights from the input to the hidden layer : learning rate * error at the hidden layer * input layer node value
		//for each connection between the input and hidden layers
	//For each neuron in the layer...
	for (int n = 0; n < layersVector[0].neuronVector.size(); ++n)
	{
		//double alpha = ((double)rand() / (RAND_MAX)); //alpha value needs to be between 0-1
		
		//For each weight in the neuron...
		for (int w = 0; w < layersVector[0].neuronVector[n].weightVector.size(); ++w)
		{
			double momentum = alpha * layersVector[0].neuronVector[n].previousDeltas[w]; // calc momentum with alpha and previous weight
			double deltaWeight = (_lRate * layersVector[0].neuronVector[n].n_error * inputsLayer[w]) + momentum;
			layersVector[0].neuronVector[n].previousDeltas[w] = deltaWeight; //store the delta weight for next back propagation

			layersVector[0].neuronVector[n].weightVector[w] += deltaWeight;
		}
	}
}

/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(std::vector<double> desiredOutput)
{
	
	double sum = 0;
	for (int i = 0; i < _outputLayerSize; ++i)
	{
		double err = desiredOutput[i] - outputsVector[i];
		sum += err*err;
	}
	return sum/_outputLayerSize;
}

/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/
void CNeuralNet::train(std::vector<std::vector<double>> inputs,	std::vector<std::vector<double>> outputs, uint trainingSetSize)
{
	std::cout << "Hidden Layer Size = " << _hiddenLayerSize << std::endl;
	std::cout << "MSE Cut Off = " << _mse_cutoff << std::endl;
	std::cout << "Learning Rate = " << _lRate << std::endl;
	
	std::cout << "\n======================================\n	--> BEGIN TRAINING <--	\n======================================\n" << std::endl;

	//for each training pattern --> until the MSE becomes suitably small 
	while (MSE > _mse_cutoff)
	{
		//For each training input...
		for (int i = 0; i < trainingSetSize; ++i)
		{
			//Feed Forward//
			feedForward(inputs[i]);

			//Propagate Backwards//
			propagateErrorBackward(outputs[i], inputs[i]);

			// Update the MSE
			MSE = meanSquaredError(outputs[i]);

			//If MSE_cutoff is met while in middle of training set, 
			//then the ANN has learnt...
			//if (MSE < _mse_cutoff)
				//break;
		}
		std::cout << "The training loop MSE is:   " << MSE << std::endl;
	}
	std::cout << "\n======================================\n	--> TRAINING COMPLETE <--	\n======================================\n" << std::endl;
}

/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
uint CNeuralNet::classify(std::vector<double> input)
{
	//Send the input to the feed forward method
	feedForward(input);

	int returnIndex = 0;

	for (int i = 0; i < outputsVector.size(); ++i)
	{
		if (outputsVector[i] > outputsVector[returnIndex])
			returnIndex = i;
	}

	
	return returnIndex;
}

/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const{
	return outputsVector[index]; 
}