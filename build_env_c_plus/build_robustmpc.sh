c++ -O4 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) robust_mpc.cc -o robustmpc$(python3-config --extension-suffix)
cp robustmpc*.so ../sim_env
echo 'Done'