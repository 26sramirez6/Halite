#include <torch/torch.h>
#include <iostream>
#include "dqn.hpp"

int main() {
    torch::Tensor x = torch::randn({5, 2, 11, 11});
    auto W = torch::randn({1,4,2,9,9,9});
    //std::cout << x << std::endl;
    x = x.unfold(2, 3, 1).unfold(3, 3, 1);
    x = x.contiguous().view({5,2,9,9,-1});
    auto out = (x.unsqueeze(1)*W).sum({2,-1});
    
    torch::Tensor y = torch::randn({10,2});
    torch::Tensor o = torch::ones({5,2});
    y = torch::cat({y, o}, 0);
    
    DQN dqn(2, 11, 6);
    std::cout << dqn << std::endl;
 }
