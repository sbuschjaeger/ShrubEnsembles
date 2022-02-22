#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <functional>

#include "Losses.h"
#include "OnlineShrubEnsemble.h"
#include "MAShrubEnsemble.h"
#include "GAShrubEnsemble.h"
#include "Serialization.h"

std::vector<std::vector<data_t>> random_data(unsigned int N, unsigned int d) {
	auto gen = std::bind(std::uniform_real_distribution<>(0,1),std::default_random_engine());
	std::vector<data_t> tmp(N*d);
	std::generate(tmp.begin(), tmp.end(), gen);

	std::vector<std::vector<data_t>> X(N);
	for (unsigned int i = 0; i < N; ++i) {
		X[i] = std::vector<data_t>(&tmp[i*d],&tmp[i*d] + d); 
	}
	return X;
}

std::vector<unsigned int> random_targets(unsigned int N) {
	std::vector<unsigned int> Y(N);
	auto gen = std::bind(std::uniform_int_distribution<>(0,1),std::default_random_engine());
	std::generate(Y.begin(), Y.end(), gen);
	return Y;
}

int main() {
	auto X = random_data(1 << 10, 32);
	auto Y = random_targets(1 << 10);

	auto n_classes = 2;
	auto max_depth = 15; 
	auto max_features = (int) X[0].size() / 2;
	auto seed = 1234L;
	auto step_size = 1e-3;

	auto normalize_weights = true;
	auto burnin_steps = 5;
	auto ensemble_regularizer = ENSEMBLE_REGULARIZER::from_string("hard-L0");
	auto l_ensemble_reg = 32;
	auto tree_regularizer = TREE_REGULARIZER::from_string("none");
	auto l_tree_reg = 0;
	auto epochs = 20;
	auto batch_size = (unsigned int)X.size() / 8;
	auto bootstrap = true;
	auto n_rounds = 5;
	auto n_batches = 8;

	ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,OPTIMIZER::OPTIMIZER_TYPE::NONE, TREE_INIT::TRAIN> se(
		n_classes,
		max_depth,
		seed,
		normalize_weights,
		burnin_steps,
		max_features,
		step_size,
		ensemble_regularizer,
		l_ensemble_reg,
		tree_regularizer,
		l_tree_reg
	);

	se.fit_gd(X,Y,l_ensemble_reg,bootstrap,batch_size,n_rounds,n_batches)

	auto in = serialize(se);
	auto se_new = deserialize(in);
}