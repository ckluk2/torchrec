#include <iostream>
#include <memory>
#include <string>

// #include <folly/futures/Future.h>
// #include <folly/io/IOBuf.h>
// #include <folly/json/json.h>
// #include <gflags/gflags.h>
// #include <glog/logging.h>
// #include <grpc++/grpc++.h>
// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/grpcpp.h>

#include <torch/script.h>
#include <torch/nn/functional/activation.h>

int main(int argc, char* argv[]) {

	if (argc != 2) {
        std::cerr << "usage: ts-infer <path-to-exported-model>\n";
        return -1;
    }

	std::cout << "Loading model...\n";

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }

	torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval(); // turn off dropout and other training-time layers/functions

    c10::Dict<std::string, at::Tensor> dict;
    dict.insert("float_features", torch::ones({1, 13}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)));
    dict.insert("id_list_features.lengths", torch::ones({26}, torch::dtype(torch::kLong).device(torch::kCUDA, 0)));
    dict.insert("id_list_features.values", torch::ones({26}, torch::dtype(torch::kLong).device(torch::kCUDA, 0)));

    std::vector<c10::IValue> input;
    input.push_back(c10::IValue(dict));

    // Execute the model and turn its output into a tensor.
    c10::IValue output = module.forward(input).toGenericDict().at("default");
    std::cout << " Model Forward Completed, Output: " << output << std::endl;
	return 0;
}
