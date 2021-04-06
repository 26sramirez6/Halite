#pragma once
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <EigenRand/EigenRand>
#include "board.hpp"

template<typename Config>
struct BoardStore {
	static constexpr unsigned half = (Config::size / 2) + 1;
	static constexpr unsigned fourth = half / 4;
	using HalfMatI = Eigen::Array<int, half*Config::board_count, half>;
	using HalfMatF = Eigen::Array<float, half*Config::board_count, half>;
	using FourthMatF = Eigen::Array<float, fourth*Config::board_count, fourth>;
	inline void generate_boards() {
		srand(time(NULL));
		Eigen::Rand::Vmt19937_64 urng{ 0 };
		Eigen::Rand::ExtremeValueGen<float> gumbel1_gen(0, 300);
		Eigen::Rand::ExtremeValueGen<float> gumbel2_gen(0, 500);
		Eigen::Rand::BinomialGen<int> binomial_gen(1, .5);
		HalfMatF gumbel1 = gumbel1_gen.generate<HalfMatF>(half*Config::board_count, half, urng);
		FourthMatF gumbel2 = gumbel2_gen.generate<FourthMatF>(fourth*Config::board_count, fourth, urng);
		HalfMatI binomial = binomial_gen.generate<HalfMatI>(half*Config::board_count, half, urng);

		const auto quartile_add = gumbel1.max(0) * binomial.template cast<float>();
		
		for (int i = 0; i < Config::board_count; ++i) {
			m_boards[i].populate(
				quartile_add.template middleRows<half>(i*half),
				(gumbel2.template middleRows<fourth>(i*fourth)).max(0));
		}
	}



	Board<Config>				m_boards[Config::board_count];
	unsigned					m_current_index = 0;
};
