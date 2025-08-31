c++ -O4 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) env_rl.cc -o libmpccorerl$(python3-config --extension-suffix)
cp libmpccorerl*.so ../sim_env
echo 'Done'