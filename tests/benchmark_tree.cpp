#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <functional>

#include "Losses.h"
#include "DecisionTreeV2.h"
#include "GASE.h"
#include "MASE.h"
#include "OSE.h"

#include <valgrind/callgrind.h>

void print_progress(unsigned int cur_epoch, unsigned int max_epoch, data_t progress, std::string const & pre_str, unsigned int width = 100, unsigned int precision = 8) {
    //data_t progress = data_t(cur_idx) / data_t(max_idx);

    std::cout << "[" << cur_epoch << "/" << max_epoch << "] " << std::setprecision(precision) << pre_str <<  " " ;
    unsigned int pos = width * progress;
    for (unsigned int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "█";
        //else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << " " << int(progress * 100.0) << " %\r";
    std::cout << std::flush;
}

void print_matrix(std::vector<std::vector<data_t>> const &X) {
	for (auto const & Xi: X) {
	    for (auto const & Xij : Xi) {
	        std::cout << Xij << " ";
	    }
	    std::cout << std::endl;
	}
}

void print_vector(std::vector<data_t> const &X) {
	for (auto const & Xi: X) {
		std::cout << Xi << " ";
	}
	std::cout << std::endl;
}

// template <TREE_INIT tree_init, TREE_NEXT tree_next, typename internal_t>
// void print_tree(Tree<tree_init,tree_next,internal_t> &tree) {
// 	unsigned int i = 0;
// 	for (auto & n : tree.nodes) {
// 		if (n.left_is_leaf)
// 		if (n.left == 0) {
// 			std::cout << "[" << i << "] - if x[" << n.feature << "] <= " << n.threshold << "=> " << n.prediction << " else " << n.right << std::endl;
// 		} else {
// 			std::cout << "[" << i << "] - if x[" << n.feature << "] <= " << n.threshold  << "=> " << n.left << " else " << n.right << std::endl;
// 		}
// 		++i;
// 	}
// }

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

internal_t accuracy_score(std::vector<std::vector<internal_t>> const &proba, std::vector<unsigned int> const &Y) {

	internal_t accuracy = 0;
	for (unsigned int i = 0; i < proba.size(); ++i) {
		auto max_idx = std::distance(proba[i].begin(), std::max_element(proba[i].begin(), proba[i].end()));
		if (max_idx == Y[i]) {
			accuracy++;
		}
	}
	return accuracy / proba.size() * 100.0;
}

std::vector<std::vector<data_t>> X = random_data(50000, 54);
std::vector<unsigned int> Y = random_targets(50000);


int main() {
	// auto X = random_data(1 << 14, 32);
	// auto Y = random_targets(1 << 14);

	auto n_classes = 2;
	auto max_depth = 0; 
	auto max_features = 0;
	auto seed = 12345L;
	auto step_size = 0;

	DecisionTreeClassifierV2 dt(max_depth,n_classes,max_features,seed,step_size,"train","none");

    std::cout << "Generated random data with X = " << X.size() << " / " << X[0].size() << " and Y = " << Y.size() << std::endl;

	auto start = std::chrono::steady_clock::now();
	dt.fit(X,Y);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> runtime_seconds = end-start;
    
	std::cout << "=== Testing single DT ===" << std::endl;
	std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
    std::cout << "Size is " << dt.num_bytes() << " bytes" << std::endl; 
    std::cout << "Number of nodes was " << dt.num_nodes() << std::endl; 
    CALLGRIND_START_INSTRUMENTATION;
    double acc = accuracy_score(dt.predict_proba(X), Y);
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
    // double acc = 0;
    // for (unsigned int i = 0; i < 1000; ++i) {
    //     acc += accuracy_score(dt.predict_proba(X), Y);
    // }
	std::cout << "Accuracy is: " << acc << std::endl;
	std::cout << "=== Testing single DT done ===" << std::endl << std::endl;
}
	