#pragma once
#include "cdisccontroller.h"
#include "CParams.h"
#include "CDiscCollisionObject.h"
#include <cmath>

typedef unsigned int uint;

struct Action
{
	//Constructor for Action
	Action(ROTATION_DIRECTION a); //Rotation direction is enum in CDiscMineSweeper
	
	ROTATION_DIRECTION action;  //store action move to next state

	// Value for state, action pair initialized to 0
	int stateValue = 0;
		
};

struct State
{
	//State contructor
	State();

	//Position of the state on the grid
	int xPos;
	int yPos;

	// Vector of actions to be performed in the state
	std::vector<Action> stateAction;
};

struct Sweeper
{
	//Sweeper constructor
	Sweeper();

	int nextAction = 0;
	State * currentState = NULL;
	//2D vector representing the qTable
	std::vector<std::vector<State>> qTable;
};

class CQLearningController :
	public CDiscController
{
private:
	uint _grid_size_x;
	uint _grid_size_y;

	std::vector<Sweeper> sweepersVector; //vecotr of all the sweepers tables

	//REWARDS//
	int mineReward = 100;
	int supermineReward = -100;
	int rockReward = -100;
	int emptyBlockReward = 0;

	//FOR Q FUNCTION
	double learningRate = 0.25;
	double discountFactor = 0.8;
	double epsilon = 0.5;


public:
	CQLearningController(HWND hwndMain);
	virtual void InitializeLearningAlgorithm(void);
	double R(uint x, uint y, uint sweeper_no);
	void clearState(uint x, uint y, uint sweeper_no);
	virtual bool Update(void);
	int highestHistoricReturn(State * currentState, bool findMax);
	virtual ~CQLearningController(void);
};

