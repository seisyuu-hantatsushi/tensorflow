#include <stdio.h>
#include <iostream>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

int main(int argc, char **argv){
	namespace tf = tensorflow;
	std::cout << "example of mul matrix in tensorflow" << std::endl;
	tf::Scope root = tf::Scope::NewRootScope();
	tf::Status status;

	{
		tf::Output A  = tf::ops::Const(root, {{1.0f, 1.0f},{1.0f, 2.0f}});
		tf::Output P  = tf::ops::Placeholder(root, tf::DT_FLOAT);
		tf::Output m  = tf::ops::MatMul(root, A, P);

		std::vector<tf::Tensor> outputs;
		tf::ClientSession session(root);

		status = session.Run({{P,{{2.0f, 1.0f},{1.0f, 2.0f}}}},{m},&outputs);

		if(status.ok()){
			std::cout << "size of outputs: " << outputs.size() << std::endl;
			std::cout << "num of element:" <<  outputs[0].NumElements() << std::endl;
			std::cout << outputs[0].matrix<float>()(0) << ',' << outputs[0].matrix<float>()(1) << std::endl;
			std::cout << outputs[0].matrix<float>()(2) << ',' << outputs[0].matrix<float>()(3) << std::endl;
		}
		else {
			printf("Error: %s\n", status.ToString().c_str());
		}
	}

	root.ClearColocation();
	{
		tf::Output A  = tf::ops::Const(root, {{1.0f, 1.0f, 2.0f},{1.0f, 2.0f, 1.0f},{2.0f, 1.0f, 1.0f}});
		tf::Output B  = tf::ops::Const(root, {{2.0f, 1.0f, 1.0f},{1.0f, 2.0f, 1.0f},{1.0f, 1.0f, 2.0f}});
		tf::Output m  = tf::ops::MatMul(root, B, A);

		std::vector<tf::Tensor> outputs;
		tf::ClientSession session(root);

		status = session.Run({m},&outputs);

		if(status.ok()){
			std::cout << "size of outputs: " << outputs.size() << std::endl;
			std::cout << "num of element:" <<  outputs[0].NumElements() << std::endl;
			printf("[ [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(0), outputs[0].matrix<float>()(1), outputs[0].matrix<float>()(2));
			printf("  [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(3), outputs[0].matrix<float>()(4), outputs[0].matrix<float>()(5));
			printf("  [ %.5f, %.5f, %.5f] ]\n", outputs[0].matrix<float>()(6), outputs[0].matrix<float>()(7), outputs[0].matrix<float>()(8));
		}
		else {
			printf("Error: %s\n", status.ToString().c_str());
		}
	}

	root.ClearColocation();
	{
		tf::Output CA  = tf::ops::Const(root, {{1.0f, 1.0f, 2.0f},{1.0f, 2.0f, 1.0f},{2.0f, 1.0f, 1.0f}});
		tf::Output CB  = tf::ops::Const(root, {{2.0f, 1.0f, 1.0f},{1.0f, 2.0f, 1.0f},{1.0f, 1.0f, 2.0f}});
		tf::Output VA  = tf::ops::Variable(root, {3,3}, tf::DT_FLOAT);
		tf::Output initVA = tf::ops::Assign(root, VA, CA);
		tf::Output m = tf::ops::MatMul(root, VA, CB);
		tf::ClientSession session(root);
		std::vector<tf::Tensor> outputs;

		(void)session.Run({initVA}, &outputs);

		status = session.Run({m}, &outputs);

		if(status.ok()){
			std::cout << "size of outputs: " << outputs.size() << std::endl;
			std::cout << "num of element:" <<  outputs[0].NumElements() << std::endl;
			printf("[ [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(0), outputs[0].matrix<float>()(1), outputs[0].matrix<float>()(2));
			printf("  [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(3), outputs[0].matrix<float>()(4), outputs[0].matrix<float>()(5));
			printf("  [ %.5f, %.5f, %.5f] ]\n", outputs[0].matrix<float>()(6), outputs[0].matrix<float>()(7), outputs[0].matrix<float>()(8));
		}
		else {
			printf("Error: %s\n", status.ToString().c_str());
		}
	}

	return 0;
}
