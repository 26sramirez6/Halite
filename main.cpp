#include "board_store.hpp"
#include "board_config.hpp"
#include <iostream>
int main() {
	
	BoardStore<BoardConfig> board_store;
	board_store.generate_boards();
	std::cout << "boards generated" << std::endl;

    auto& board = board_store.m_boards[1];
    board.printBoard();
    std::vector<unsigned> ship_actions = {4};
    std::vector<unsigned> shipyard_actions;
    board.setActions(ship_actions, shipyard_actions);
    std::cout << "action set" << std::endl;
    board.step();
    std::cout << "step complete" << std::endl;
    board.printBoard();
	return 0;
}
