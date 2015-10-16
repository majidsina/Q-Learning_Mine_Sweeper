/**
         (                                      
   (     )\ )                                   
 ( )\   (()/(   (    ) (        (        (  (   
 )((_)   /(_)) ))\( /( )(   (   )\  (    )\))(  
((_)_   (_))  /((_)(_)|()\  )\ |(_) )\ )((_))\  
 / _ \  | |  (_))((_)_ ((_)_(_/((_)_(_/( (()(_) 
| (_) | | |__/ -_) _` | '_| ' \)) | ' \)) _` |  
 \__\_\ |____\___\__,_|_| |_||_||_|_||_|\__, |  
                                        |___/   

Refer to Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
for a detailed discussion on Q Learning
*/
#include "CQLearningController.h"

/////////////////////////////////
//CONSTRUCTORS FOR THE Q TABLE//
///////////////////////////////
Action::Action(ROTATION_DIRECTION a) : action(a){}

State::State()
{
	stateAction.push_back(Action(NORTH)); // = 1
	stateAction.push_back(Action(SOUTH)); // = 3
	stateAction.push_back(Action(EAST)); // = 0
	stateAction.push_back(Action(WEST)); // = 2
}

Sweeper::Sweeper(){}
/////////////////////////////////////////////////////


CQLearningController::CQLearningController(HWND hwndMain):
	CDiscController(hwndMain),
	_grid_size_x(CParams::WindowWidth / CParams::iGridCellDim + 1),
	_grid_size_y(CParams::WindowHeight / CParams::iGridCellDim + 1)
{
}
/**
 The update method should allocate a Q table for each sweeper (this can
 be allocated in one shot - use an offset to store the tables one after the other)

 You can also use a boost multiarray if you wish
*/
void CQLearningController::InitializeLearningAlgorithm(void)
{
	//For each sweeper...
	for (int i = 0; i < CParams::iNumSweepers; i++)
	{
		Sweeper tempSweeper;

		//For each x coord (column)...
		for (int j = 0; j < _grid_size_x; j++)
		{
			//create a vector for a column of the qTable
			std::vector<State> qTableCol;

			//For each y coord (row)...
			for (int k = 0; k < _grid_size_y; k++)
			{
				//Add state struct to each row in the column
				qTableCol.push_back(State());
			}
			//Add the column to the sweepers Q table
			tempSweeper.qTable.push_back(qTableCol);
		}
		//Add the temporary sweeper to the vecotr of sweepers
		sweepersVector.push_back(tempSweeper);
	}
}
/**
 The immediate reward function. This computes a reward upon achieving the goal state of
 collecting all the mines on the field. It may also penalize movement to encourage exploring all directions and 
 of course for hitting supermines/rocks!
*/
double CQLearningController::R(uint x,uint y, uint sweeper_no){
	
	bool checkObj = false;

	//Loop through all the objects to see if one is at the position of the sweeper..
	for (int i = 0; i < m_vecObjects.size(); ++i)
	{
		// get the x, y coordinates of the object on the grid
		int xPos = m_vecObjects[i]->getPosition().x / CParams::iGridCellDim;
		int yPos = m_vecObjects[i]->getPosition().y / CParams::iGridCellDim;

		//If the corrdinates of the object is the same as the sweeper...
		if (xPos == x && yPos == y)
		{
			//MINE
			if (m_vecObjects[i]->getType() == CCollisionObject::Mine)
			{
				clearState(x, y, sweeper_no);
				return mineReward;
			}
			//ROCK
			else if (m_vecObjects[i]->getType() == CCollisionObject::Rock)
			{
				return rockReward;
			}
			//SUPER MINE
			else if (m_vecObjects[i]->getType() == CCollisionObject::SuperMine)
			{
				clearState(x, y, sweeper_no);
				return supermineReward;
			}

			checkObj = true; // object at position of sweeper
		}
	}

	//if no object is found in the position of the sweeper
	if (!checkObj)
	{
		return emptyBlockReward;
	}

	return 0;
}

/**
This method is used to clear the state value of a specific block in the q table if a mine/supermine is found on the block.
Mine/supermine won't be at this position in the future - so stop from moving here
*/
void CQLearningController::clearState(uint x, uint y, uint sweeper_no)
{
	sweepersVector[sweeper_no].qTable[x][y].stateAction[0].stateValue = 0;
	sweepersVector[sweeper_no].qTable[x][y].stateAction[1].stateValue = 0;
	sweepersVector[sweeper_no].qTable[x][y].stateAction[2].stateValue = 0;
	sweepersVector[sweeper_no].qTable[x][y].stateAction[3].stateValue = 0;
}

The update method. Main loop body of our Q Learning implementation
See: Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
*/
bool CQLearningController::Update(void)
{
	//m_vecSweepers is the array of minesweepers
	//everything you need will be m_[something] ;)
	uint cDead = std::count_if(m_vecSweepers.begin(),
							   m_vecSweepers.end(),
						       [](CDiscMinesweeper * s)->bool{
								return s->isDead();
							   });
	if (cDead == CParams::iNumSweepers){
		printf("All dead ... skipping to next iteration\n");
		m_iTicks = CParams::iNumTicks;
	}

	for (uint sw = 0; sw < CParams::iNumSweepers; ++sw){
		if (m_vecSweepers[sw]->isDead()) continue;
		/**
		Q-learning algorithm according to:
		Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
		*/
		//1:::Observe the current state:
		//TODO
		//2:::Select action with highest historic return:
		//TODO
		//now call the parents update, so all the sweepers fulfill their chosen action
	}
	
	CDiscController::Update(); //call the parent's class update. Do not delete this.
	
	for (uint sw = 0; sw < CParams::iNumSweepers; ++sw){
		if (m_vecSweepers[sw]->isDead()) continue;
		//TODO:compute your indexes.. it may also be necessary to keep track of the previous state
		//3:::Observe new state:
		//TODO
		//4:::Update _Q_s_a accordingly:
		//TODO
	}
	return true;
}

CQLearningController::~CQLearningController(void)
{
	//TODO: dealloc stuff here if you need to	
}
