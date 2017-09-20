#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

A = tf.constant([[1.0,1.0],[1.0,1.0]], dtype=tf.float32)
B = tf.constant([[2.0,1.0],[1.0,2.0]], dtype=tf.float32)

session = tf.Session()
r = session.run(tf.add(A,B))

print (r)

