#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>

double median(std::vector<double> times)
{
  size_t size = times.size();
  std::sort(times.begin(), times.end());
  if (size % 2 == 0)
  {
    return (times[size / 2 - 1] + times[size / 2]) / 2;
  }
  else
  {
    return times[size / 2];
  }
}

int main(int argc, const char *argv[])
{
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  torch::Device device = torch::kCPU;
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  module.to(device);
  std::cout << "Model loaded\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({20, 13, 1}, torch::kFloat64).to(device));

  int N_ITERS = 100;
  std::vector<double> times;
  for (int i = 0; i < N_ITERS; ++i)
  {
    auto start = std::chrono::high_resolution_clock::now();
    at::Tensor output = module.forward(inputs).toTensor();
    auto stop = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e6;
    times.push_back(time);
    std::cout << "eval time " << i << ": " << time << " seconds" << std::endl;
  }

  std::sort(times.begin(), times.end());
  double med = (times.size() % 2 == 0) ? (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2 : times[times.size() / 2];
  std::cout << "median: " << med << " seconds" << std::endl;
}
