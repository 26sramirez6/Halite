#pragma once

struct BoardConfig {
	static constexpr unsigned size = 11;
	static constexpr double starting_halite = 5000.;
	static constexpr double convert_cost = 11.;
	static constexpr int random_seed = -1;
	static constexpr double spawn_cost = 500.;
	static constexpr double move_cost = 0.;
	static constexpr double collect_rate = 0.25;
	static constexpr double regen_rate = 0.02;
	static constexpr double max_cell_halite = 500.;
	static constexpr unsigned board_count = 500;
};

