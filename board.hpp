#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>

#include "ship.hpp"
#include "board_config.hpp"


static inline int rand_in_range(int lower, int upper) {
	return (rand() % (upper - lower + 1)) + lower;
}


template<typename Config>
struct Board {
	static constexpr unsigned half = (Config::size / 2) + 1;
	static constexpr unsigned fourth = half / 4;
	static constexpr unsigned p0_starting_index = Config::size * (Config::size / 2) + (Config::size / 2);
	using GridMat = Eigen::Array<float, Config::size, Config::size>;
	using QuartileMatF = Eigen::Array<float, half, half>;
	using CornerMatF = Eigen::Array<float, fourth, fourth>;
	using QuartileMatI = Eigen::Array<int, half, half>;
	Board() : 
		m_step(0), 
		m_p0_halite(Config::starting_halite),
		m_halite(Config::size*Config::size, 0), 
		m_ships(),
		m_shipyards(),
		m_has_ship(Config::size, 0),
		m_has_shipyard(Config::size, 0) {}

	inline void populate(
		const Eigen::Ref<const QuartileMatF>& _quartile_add,
		const Eigen::Ref<const CornerMatF>& _corner_add) {

		QuartileMatF quartile = QuartileMatF::Zero();
		for (int i = 0; i < half; ++i) {
			quartile(rand() % (half - 1), rand() % (half - 1)) = i * i;
			quartile(rand_in_range(half / 2, half - 1), rand_in_range(half / 2, half - 1)) = i * i;
		}

		for (int c = 0; c < half; ++c) {
			for (int r = 0; r < half; ++r) {
				const float value = quartile(r, c);
				if (value == 0) {
					continue;
				}

				const int radius = std::min(static_cast<int>(std::round(std::sqrt(value / half))), 1);
				for (int c2 = c - radius + 1; c2 < c + radius; ++c2) {
					for (int r2 = r - radius + 1; r2 < r + radius; ++r2) {
						const float distance = std::sqrt(std::pow(std::abs(r2 - r), 2) + std::pow(std::abs(c2 - c), 2));
						quartile(r2, c2) += std::pow(value / std::max(1.f, distance), distance);
					}
				}
			}
		}

		quartile += _quartile_add;
		quartile.bottomRightCorner<fourth, fourth>() += _corner_add;
		const float quartile_sum = quartile.sum();
		const float multiplier = Config::starting_halite / quartile_sum / 4.;
		quartile *= multiplier;

		 
		for (int c = 0; c < half; ++c) {
			for (int r = 0; r < half; ++r) {
				m_halite[Config::size*r + c] = quartile(r, c);
				m_halite[Config::size*r + (Config::size - c + 1)] = quartile(r, c);
				m_halite[Config::size * (Config::size - 1) - (Config::size * r) + c] = quartile(r, c);
				m_halite[Config::size * (Config::size - 1) - (Config::size * r) + (Config::size - c - 1)] = quartile(r, c);
			}
		}

		Ship p0_start_ship;
		index_to_point(p0_starting_index, p0_start_ship.x, p0_start_ship.y);
		m_ships.emplace_back(p0_start_ship);
	}

	static inline unsigned point_to_index(uint8_t x, uint8_t y) {
		return (Config::size - y - 1) * Config::size + x;
	}

	static inline void index_to_point(const unsigned _index, uint8_t& x_, uint8_t& y_) {
		auto dv = std::div(static_cast<int>(_index), static_cast<int>(Config::size));
		x_ = dv.rem;
		y_ = Config::size - dv.quot - 1;
	}

	inline void step() {
		for (auto& ship : m_ships) {
			const unsigned index = point_to_index(ship.x, ship.y);
			switch (ship.action) {
			case ShipAction::NONE:
				const float delta = m_halite[index] * Config::collect_rate;
				m_halite[index] -= delta;
				ship.cargo += delta;
				break;
			case ShipAction::CONVERT:
				m_p0_halite -= Config::convert_cost;
				m_shipyards.emplace_back(Shipyard{ship.x, ship.y, ShipyardAction::NONE});
				m_has_ship[index] = false;
				m_has_shipyard[index] = true;
				break;
			case ShipAction::MOVE_NORTH:
				ship.y = (ship.y == Config::size - 1) ? 0 : ship.y + 1;
				m_has_ship[index] = false;
				m_has_ship[point_to_index(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_EAST:
				ship.x = (ship.x == Config::size - 1) ? 0 : ship.x + 1;
				m_has_ship[index] = false;
				m_has_ship[point_to_index(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_SOUTH:
				ship.y = (ship.y == 0) ? Config::size - 1 : ship.y - 1;
				m_has_ship[index] = false;
				m_has_ship[point_to_index(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_WEST:
				ship.x = (ship.x == 0) ? Config::size - 1 : ship.x - 1;
				m_has_ship[index] = false;
				m_has_ship[point_to_index(ship.x, ship.y)] = true;
				break;
			}
			ship.action = ShipAction::NONE;
			

			if (ship.x == m_shipyards[0].x && ship.y == m_shipyards[0].y) {
				m_p0_halite += ship.cargo;
				ship.cargo = 0;
			}
		}

		for (auto& shipyard : m_shipyards) {
			if (shipyard.action == ShipyardAction::SPAWN) {
				m_p0_halite -= Config::spawn_cost;
				m_has_ship[point_to_index(shipyard.x, shipyard.y)] = true;
				m_ships.emplace_back(Ship{ shipyard.x, shipyard.y, ShipAction::NONE });
			}
			shipyard.action = ShipyardAction::NONE;
		}

		for (int i = 0; i < Config::size * Config::size; ++i) {
			m_halite[i] *= (1 + Config::regen_rate);
		}
	}

	unsigned						m_step;
	int								m_p0_halite;
	std::vector<float>				m_halite; 
	std::vector<Ship>				m_ships;
	std::vector<Shipyard>			m_shipyards;
	std::vector<bool>				m_has_ship;
	std::vector<bool>				m_has_shipyard;
};

