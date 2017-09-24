#include <stdio.h>
#include <iostream>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

int main(int argc, char **argv){
	namespace tf = tensorflow;
	std::cout << "example of loop flow in tensorflow" << std::endl;
	tf::Scope root = tf::Scope::NewRootScope();
	tf::Status status;

	// changing the value added by increment value in placeholder
	{
		unsigned int i;
		tf::Output V = tf::ops::Variable(root, {3,3}, tf::DT_FLOAT);
		tf::Output P = tf::ops::Placeholder(root, tf::DT_FLOAT);
		tf::Output initV = tf::ops::Assign(root, V, tf::ops::Const(root,{{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f}}));
		tf::Output update = tf::ops::Assign(root,V, tf::ops::Add(root,V,P));
		tf::ClientSession session(root);

		std::vector<tf::Tensor> outputs;

		(void)session.Run({initV},&outputs);

		for(i=1;i<=10;i++){
			tf::Tensor incTensor = tf::Tensor(tf::DT_FLOAT, {3,3});

			for(unsigned int j=0; j<9; j++){
				incTensor.matrix<float>()(j) = static_cast<float>(i);
			}

			status = session.Run({{P,incTensor}},{update},&outputs);
			if(status.ok()){
				printf("[ [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(0), outputs[0].matrix<float>()(1), outputs[0].matrix<float>()(2));
				printf("  [ %.5f, %.5f, %.5f] \n",  outputs[0].matrix<float>()(3), outputs[0].matrix<float>()(4), outputs[0].matrix<float>()(5));
				printf("  [ %.5f, %.5f, %.5f] ]\n", outputs[0].matrix<float>()(6), outputs[0].matrix<float>()(7), outputs[0].matrix<float>()(8));
			}
			else{
				printf("Error: %s\n", status.ToString().c_str());
			}
		}
	}

	return 0;
}
