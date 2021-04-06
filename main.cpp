#include "board_store.hpp"
#include "board_config.hpp"
#include <iostream>
int main() {
	
	BoardStore<BoardConfig> board_store;
	board_store.generate_boards();
	std::cout << "here" << std::endl;
	return 0;
}