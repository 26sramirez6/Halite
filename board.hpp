#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <map>
#include <cstdio>
#include <sstream>
#include <iostream>
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
		m_global_ship_id(0),
        m_global_shipyard_id(0),
		m_p0_halite(Config::starting_halite),
		m_halite(Config::size*Config::size, 0), 
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
		quartile.template bottomRightCorner<fourth, fourth>() += _corner_add;
		const float quartile_sum = quartile.sum();
		const float multiplier = Config::starting_halite / quartile_sum / 4.;
		quartile *= multiplier;

		 
		for (int c = 0; c < half; ++c) {
			for (int r = 0; r < half; ++r) {
				m_halite[Config::size*r + c] = quartile(r, c);
				m_halite[Config::size*r + (Config::size - c - 1)] = quartile(r, c);
				m_halite[Config::size*(Config::size - 1) - (Config::size * r) + c] = quartile(r, c);
				m_halite[Config::size*(Config::size - 1) - (Config::size * r) + (Config::size - c - 1)] = quartile(r, c);
			}
		}

		Ship p0_start_ship{0,0,0,0.f,ShipAction::NONE};
		indexToPoint(p0_starting_index, p0_start_ship.x, p0_start_ship.y);
		m_has_ship[p0_starting_index] = true;
		m_ship_map.emplace(p0_start_ship.id, p0_start_ship);
		++m_global_ship_id;
	}

	static inline unsigned pointToIndex(uint8_t x, uint8_t y) {
		return (Config::size - y - 1) * Config::size + x;
	}

	static inline void indexToPoint(const unsigned _index, uint8_t& x_, uint8_t& y_) {
		auto dv = std::div(static_cast<int>(_index), static_cast<int>(Config::size));
		x_ = dv.rem;
		y_ = Config::size - dv.quot - 1;
	}

	inline float getPlayerHalite() { return m_p0_halite; }
	inline std::vector<float>& getHaliteGrid() { return m_halite; }
	inline bool pointHasShip(uint8_t x, uint8_t y) { return m_has_ship[pointToIndex(x, y)]; }
	inline bool pointHasShipyard(uint8_t x, uint8_t y) { return m_has_shipyard[pointToIndex(x, y)]; }
	inline bool indexHasShip(unsigned index) { return m_has_ship[index]; }
	inline bool indexHasShipyard(unsigned index) { return m_has_shipyard[index]; }
	inline unsigned getShipCount() { return m_ship_map.size(); }
	inline unsigned getShipyardCount() { return m_shipyard_map.size(); }
	inline unsigned getStep() { return m_step; }
	inline Ship& getShip(int key) { return m_ship_map.at(key); }

	inline void step() {
	    std::vector<int> remove_ships;
	    std::vector<Shipyard> add_shipyards;
		for (auto& kv : m_ship_map) {
		    auto& ship = kv.second;
			const unsigned index = pointToIndex(ship.x, ship.y);
			switch (ship.action) {
			case ShipAction::NONE:
            {
				const float delta = m_halite[index] * Config::collect_rate;
				m_halite[index] -= delta;
				ship.cargo += delta;
            }
				break;
			case ShipAction::CONVERT:
				m_p0_halite -= Config::convert_cost;
				m_has_ship[index] = false;
				m_has_shipyard[index] = true;
				remove_ships.push_back(ship.id);
				add_shipyards.emplace_back(Shipyard{m_global_shipyard_id, ship.x, ship.y, ShipyardAction::NONE});
				++m_global_shipyard_id;
				break;
			case ShipAction::MOVE_NORTH:
				ship.y = (ship.y == Config::size - 1) ? 0 : ship.y + 1;
				m_has_ship[index] = false;
				m_has_ship[pointToIndex(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_EAST:
				ship.x = (ship.x == Config::size - 1) ? 0 : ship.x + 1;
				m_has_ship[index] = false;
				m_has_ship[pointToIndex(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_SOUTH:
				ship.y = (ship.y == 0) ? Config::size - 1 : ship.y - 1;
				m_has_ship[index] = false;
				m_has_ship[pointToIndex(ship.x, ship.y)] = true;
				break;
			case ShipAction::MOVE_WEST:
				ship.x = (ship.x == 0) ? Config::size - 1 : ship.x - 1;
				m_has_ship[index] = false;
				m_has_ship[pointToIndex(ship.x, ship.y)] = true;
				break;
			}
			ship.action = ShipAction::NONE;
			
			if (m_shipyard_map.size() > 0) {
			    const auto& shipyard = m_shipyard_map.at(0);
			    if (ship.x == shipyard.x && ship.y == shipyard.y) {
                    m_p0_halite += ship.cargo;
                    ship.cargo = 0;
			    }
			}
		}

		for (auto& kv : m_shipyard_map) {
		    auto& shipyard = kv.second;
			if (shipyard.action == ShipyardAction::SPAWN) {
				m_p0_halite -= Config::spawn_cost;
				m_has_ship[pointToIndex(shipyard.x, shipyard.y)] = true;
				m_ship_map.emplace(m_global_ship_id, Ship{m_global_ship_id, shipyard.x, shipyard.y, 0.f, ShipAction::NONE });
				m_global_ship_id++;
			}
			shipyard.action = ShipyardAction::NONE;
		}

		for (auto id : remove_ships) {
            m_ship_map.erase(id);
        }

		for (const auto& shipyard : add_shipyards) {
		    m_shipyard_map.emplace(shipyard.id, shipyard);
		}

		for (int i = 0; i < Config::size * Config::size; ++i) {
			m_halite[i] *= (1 + Config::regen_rate);
		}

		m_step++;
	}

	void printBoard() {
	    std::stringstream ss;
	    for (int y = 0; y < Config::size; ++y) {
	        for (int x = 0; x < Config::size; ++x) {
	            const unsigned index = pointToIndex(x,Config::size-y-1);
	            ss << "|";
	            if (m_has_ship[index]) {
	                ss << "a";
	            } else {
	                ss << " ";
	            }

	            const int normalized_halite = static_cast<int>(9 * m_halite[index] / Config::max_cell_halite);
	            ss << normalized_halite;
	            if (m_has_shipyard[index]) {
	                ss << "A";
	            } else {
	                ss << " ";
	            }
	        }
	        ss << "|\n";
	    }
	    std::cout << ss.str() << std::endl;
	}

	void setActions(const std::vector<unsigned>& _ship_actions,
	        const std::vector<unsigned>& _shipyard_actions) {
	    assert(_ship_actions.size() == m_ship_map.size());
	    assert(_shipyard_actions.size() == m_shipyard_map.size());
	    {
            int i = 0;
            for (auto& kv : m_ship_map) {
                kv.second.action = static_cast<ShipAction>(_ship_actions[i]);
                ++i;
            }
	    }
	    {
            int i = 0;
            for (auto& kv : m_shipyard_map) {
                kv.second.action = static_cast<ShipyardAction>(_shipyard_actions[i]);
                ++i;
            }
	    }
	}

	unsigned						    m_step;
	int                                 m_global_ship_id;
	int                                 m_global_shipyard_id;
	float								m_p0_halite;
	std::vector<float>				    m_halite;
	std::vector<bool>				    m_has_ship;
	std::vector<bool>				    m_has_shipyard;
	std::map<int, Ship>                 m_ship_map;
	std::map<int, Shipyard>             m_shipyard_map;

};
