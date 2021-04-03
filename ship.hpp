#pragma once
#include <cinttypes>

enum class ShipAction {
	NONE,
	MOVE_NORTH,
	MOVE_EAST,
	MOVE_SOUTH,
	MOVE_WEST,
	CONVERT,
	Count
};

enum class ShipyardAction {
	NONE,
	SPAWN,
	Count
};

struct Ship {
	uint8_t x;
	uint8_t y;
	float cargo;
	ShipAction action;
};

struct Shipyard {
	uint8_t x;
	uint8_t y;
	ShipyardAction action;
};