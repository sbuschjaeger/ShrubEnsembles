#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <functional>

#include "Losses.h"
#include "DecisionTree.h"
#include "GASE.h"
#include "MASE.h"
//#include "TreeEnsemble.h"


#include <valgrind/callgrind.h>

// void print_progress(unsigned int cur_epoch, unsigned int max_epoch, data_t progress, std::string const & pre_str, unsigned int width = 100, unsigned int precision = 8) {
//     //data_t progress = data_t(cur_idx) / data_t(max_idx);

//     std::cout << "[" << cur_epoch << "/" << max_epoch << "] " << std::setprecision(precision) << pre_str <<  " " ;
//     unsigned int pos = width * progress;
//     for (unsigned int i = 0; i < width; ++i) {
//         if (i < pos) std::cout << "█";
//         //else if (i == pos) std::cout << ">";
//         else std::cout << " ";
//     }
//     std::cout << " " << int(progress * 100.0) << " %\r";
//     std::cout << std::flush;
// }

// void print_matrix(std::vector<std::vector<data_t>> const &X) {
// 	for (auto const & Xi: X) {
// 	    for (auto const & Xij : Xi) {
// 	        std::cout << Xij << " ";
// 	    }
// 	    std::cout << std::endl;
// 	}
// }

// void print_vector(std::vector<data_t> const &X) {
// 	for (auto const & Xi: X) {
// 		std::cout << Xi << " ";
// 	}
// 	std::cout << std::endl;
// }

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

matrix2d<data_t> random_data(unsigned int N, unsigned int d) {
	auto gen = std::bind(std::uniform_real_distribution<>(0,1),std::default_random_engine());
    matrix2d<data_t> tmp(N, d);
	//std::generate(tmp._data, tmp._data + N*d, gen);
	std::generate(tmp.begin(), tmp.end(), gen);

    return tmp;
}

matrix1d<unsigned int> random_targets(unsigned int N) {
	matrix1d<unsigned int> Y(N);
	//std::vector<unsigned int> Y(N);
	auto gen = std::bind(std::uniform_int_distribution<>(0,1),std::default_random_engine());
	std::generate(Y.begin(), Y.end(), gen);
	return Y;
}

internal_t accuracy_score(matrix2d<data_t> const &proba, matrix1d<unsigned int> const &Y) {

	internal_t accuracy = 0;
	for (unsigned int i = 0; i < proba.rows; ++i) {
		auto max_idx = std::distance(proba(i).begin(), std::max_element(proba(i).begin(), proba(i).end()));
		if (max_idx == Y(i)) {
			accuracy++;
		}
	}
	return accuracy / proba.rows * 100.0;
}

matrix2d<data_t> X = random_data(5000, 54);
matrix1d<unsigned int> Y = random_targets(5000);


int main() {
	auto n_classes = 2;
	auto max_depth = 0; 
	auto max_features = 0;
	auto seed = 12345L;
	auto step_size = 0;

	DecisionTreeClassifier dt(max_depth,n_classes,max_features,seed,step_size,"train","none");

    std::cout << "Generated random data with X = " << X.rows << " / " << X.cols << " and Y = " << Y.dim << std::endl;

	auto start = std::chrono::steady_clock::now();
	dt.fit(X,Y);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> runtime_seconds = end-start;
    
	std::cout << "=== Testing single DT ===" << std::endl;
	std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
    std::cout << "Size is " << dt.num_bytes() << " bytes" << std::endl; 
    std::cout << "Number of nodes was " << dt.num_nodes() << std::endl; 
    //CALLGRIND_START_INSTRUMENTATION;
    double acc = accuracy_score(dt.predict_proba(X), Y);
    // CALLGRIND_STOP_INSTRUMENTATION;
    // CALLGRIND_DUMP_STATS;
    // double acc = 0;
    // for (unsigned int i = 0; i < 1000; ++i) {
    //     acc += accuracy_score(dt.predict_proba(X), Y);
    // }
	std::cout << "Accuracy is: " << acc << std::endl;
	std::cout << "=== Testing single DT done ===" << std::endl << std::endl;

    auto loss = "mse";
	auto optimizer = "sgd";
	auto tree_init_mode = "train";
	auto n_trees = 32;
	auto n_batches = 16;
	auto n_rounds = 5;
	auto init_batch_size = 32;
	auto bootstrap = true;
	
	GASE ga(n_classes, max_depth, seed, max_features, loss, step_size, optimizer, tree_init_mode, n_trees, n_batches, n_rounds, init_batch_size, bootstrap);
	start = std::chrono::steady_clock::now();
	ga.fit(X,Y);
	end = std::chrono::steady_clock::now();
	runtime_seconds = end-start;
    
	std::cout << "=== Testing GASE ===" << std::endl;
	std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
    std::cout << "Size is " << ga.num_bytes() << " bytes" << std::endl; 
    std::cout << "Number of nodes was " << ga.num_nodes() << std::endl; 
	std::cout << "Accuracy is: " << accuracy_score(ga.predict_proba(X), Y) << std::endl;
	std::cout << "=== Testing GASE done ===" << std::endl << std::endl;

	auto burnin_steps = 5;
	auto n_parallel = 8;
	auto batch_size = 1024;
	MASE ma(n_classes, max_depth, seed, burnin_steps, max_features, loss, step_size, optimizer, tree_init_mode, n_trees, n_parallel, n_rounds, batch_size, bootstrap);
	start = std::chrono::steady_clock::now();
	ma.fit(X,Y);
	end = std::chrono::steady_clock::now();
	runtime_seconds = end-start;
    
	std::cout << "=== Testing MASE ===" << std::endl;
	std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
    std::cout << "Size is " << ma.num_bytes() << " bytes" << std::endl; 
    std::cout << "Number of nodes was " << ma.num_nodes() << std::endl; 
	std::cout << "Accuracy is: " << accuracy_score(ma.predict_proba(X), Y) << std::endl;
	std::cout << "=== Testing MASE done ===" << std::endl << std::endl;
	
	// auto epochs = 5;
	// auto normalize_weights = true;
	// OSE oest(n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, "mse", step_size, "sgd", "train", "none", 0);

	// start = std::chrono::steady_clock::now();
	// LOSS::Loss<LOSS::TYPE::MSE> mse_loss;

    // for (unsigned int i = 0; i < epochs; ++i) {
    //     unsigned int cnt = 0;
    //     internal_t loss_epoch = 0;
    //     unsigned int nonzero_epoch = 0;
	// 	internal_t accuracy_epoch = 0;

    //     unsigned int batch_cnt = 0;
    //     while(cnt < X.size()) {
	// 		auto cur_batch_size = std::min(static_cast<int>(X.size() - cnt), static_cast<int>(batch_size));
	// 		if (cur_batch_size <= 0) break;

	// 		auto batch = sample_data(X, Y, cur_batch_size, false, cnt);
    //         auto & data = std::get<0>(batch);
    //         auto & target = std::get<1>(batch);
	// 		cnt += cur_batch_size;

	// 		auto proba = oest.predict_proba(data);
	// 		accuracy_epoch += accuracy_score(proba, target);

    //         oest.next(data, target);
	// 		std::vector<std::vector<data_t>> losses = mse_loss.loss(proba, target);
	// 		internal_t loss = mean_all_dim(losses);

    //         nonzero_epoch += oest.num_trees();
    //         loss_epoch += loss;
    //         batch_cnt++;
    //         std::stringstream ss;
    //         ss << std::setprecision(4) << "loss: " << loss_epoch / batch_cnt << " nonzero: " << int(nonzero_epoch / batch_cnt) << " acc " << (accuracy_epoch / batch_cnt);
	// 		internal_t progress = internal_t(cnt) / X.size();
    //         print_progress(i, epochs - 1, progress, ss.str() );
    //     }
    //     std::cout << std::endl;
    // }

    // end = std::chrono::steady_clock::now();
	// runtime_seconds = end-start;
    
	// std::cout << "=== Testing OSE ===" << std::endl;
	// std::cout << "Runtime was " << runtime_seconds.count() << " seconds" << std::endl; 
    // std::cout << "Size is " << oest.num_bytes() << " bytes" << std::endl; 
    // std::cout << "Number of nodes was " << oest.num_nodes() << std::endl; 
	// std::cout << "Accuracy is: " << accuracy_score(oest.predict_proba(X), Y) << std::endl;
	// std::cout << "=== Testing OSE done ===" << std::endl << std::endl;
}
	