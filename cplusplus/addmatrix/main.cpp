#include <stdio.h>
#include <iostream>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

int main(int argc, char **argv){
	namespace tf = tensorflow;
	std::cout << "example of add matrix in tensorflow" << std::endl;
	tf::Scope root = tf::Scope::NewRootScope();

	tf::Output A = tf::ops::Const(root, {{1.0f, 1.0f, 2.0f},{1.0f, 2.0f, 1.0f},{2.0f, 1.0f, 1.0f}});
	tf::Output B = tf::ops::Const(root, {{2.0f, 1.0f, 1.0f},{1.0f, 2.0f, 1.0f},{1.0f, 1.0f, 2.0f}});

	tf::Output a = tf::ops::Add(root, A, B);

	std::vector<tf::Tensor> outputs;
	tf::ClientSession session(root);

	tf::Status status;

	status = session.Run({a},&outputs);

	if(status.ok()){
		std::cout << outputs[0].matrix<float>() << std::endl;

	}
	else {
		printf("Error: %s\n", status.ToString().c_str());
	}

	return 0;
}
