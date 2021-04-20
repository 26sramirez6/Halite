#include "board_store.hpp"
#include "board_config.hpp"
#include <iostream>


template<typename Board>
void move_north(Board& board) {
    std::vector<unsigned> ship_actions = {1};
    std::vector<unsigned> shipyard_actions = {0};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_east(Board& board) {
    std::vector<unsigned> ship_actions = {2};
    std::vector<unsigned> shipyard_actions = {0};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_south(Board& board) {
    std::vector<unsigned> ship_actions = {3};
    std::vector<unsigned> shipyard_actions = {0};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_west(Board& board) {
    std::vector<unsigned> ship_actions = {4};
    std::vector<unsigned> shipyard_actions = {0};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_none(Board& board) {
    std::vector<unsigned> ship_actions = {0};
    std::vector<unsigned> shipyard_actions = {0};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_convert(Board& board) {
    std::vector<unsigned> ship_actions = {5};
    std::vector<unsigned> shipyard_actions;
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

template<typename Board>
void move_spawn(Board& board) {
    std::vector<unsigned> ship_actions;
    std::vector<unsigned> shipyard_actions = {1};
    board.setActions(ship_actions, shipyard_actions);
    board.step();
    board.printBoard();
}

void assert_near(float expected, float test, double threshold) {
    std::cout << expected << ", " << test << std::endl;
    assert(fabs(expected - test) <= threshold);
}

int main() {
	BoardStore<BoardConfig> board_store;
	board_store.generate_boards();
	std::cout << "boards generated" << std::endl;
	constexpr double threshold = 1e-4;
    auto& board = board_store.m_boards[1];
    assert_near(5000, board.getPlayerHalite(), threshold);
    move_convert(board);
    assert_near(4500, board.getPlayerHalite(), threshold);
    move_spawn(board);
    assert_near(4000, board.getPlayerHalite(), threshold);
    move_north(board);
    assert_near(0, board.getShip(1).cargo, threshold);
    move_none(board);
    assert_near(48.1757, board.getShip(1).cargo, threshold);
    move_south(board);
    assert_near(4048.1757, board.getPlayerHalite(), threshold);
    std::cout << "completed test" << std::endl;
	return 0;
}
