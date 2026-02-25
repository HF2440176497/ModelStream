#!/bin/bash

# 切换到build目录并执行make，然后切换到bin目录
cd ../build && cmake .. && make && cd ../bin
