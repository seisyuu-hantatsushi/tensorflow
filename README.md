# recipes of tensorflow
## How to install tensorflow library of C++
1. How to build tensorflow library of C++.

```
$ bazel build -c opt --copt=-march=native tensorflow:libtensorflow.so
$ bazel build -c opt --copt=-march=native tensorflow:libtensorflow_cc.so
```

1. Copy tensorflow library to install directory.

```
$ cp bazel-bin/tensorflow/libtensorflow.so <install path>
$ cp bazel-bin/tensorflow/libtensorflow_cc.so <install path>
```

1. Copy headers of tensorflow library.

```
$ cp <python package lib path>/tensorflow <install path of headers>
$ cp tensorflow/cc <install path of headers>
```

