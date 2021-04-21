#ifndef DQN_HPP
#define DQN_HPP


struct LocallyConnected2DImpl : torch::nn::Module {
    LocallyConnected2DImpl(
            const int64_t _in_channels, 
            const int64_t _out_channels, 
            const int64_t _output_size0,
            const int64_t _output_size1,
            const int64_t _kernel_size, 
            const int64_t _stride) : 
        m_in_channels(_in_channels),
        m_out_channels(_out_channels),
        m_kernel(_kernel_size), 
        m_stride(_stride) {
        m_W = register_parameter("W", torch::randn({1, _out_channels, _in_channels, _output_size0, _output_size1, _kernel_size*_kernel_size}));
        m_b = register_parameter("b", torch::randn({1, _out_channels, _output_size0, _output_size1}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto batches = x.size(0);
        x = x.unfold(2, m_kernel, m_stride).unfold(3, m_kernel, m_stride);
        x = x.contiguous().view({batches, m_in_channels, m_output_size0, m_output_size1, -1});
        auto out = (x.unsqueeze(1) * m_W).sum({2,-1});
        out = out + m_b;
        return out;
    }
    
    int64_t m_in_channels;
    int64_t m_out_channels;
    int64_t m_output_size0;
    int64_t m_output_size1;
    int64_t m_kernel;
    int64_t m_stride;
    torch::Tensor m_W;
    torch::Tensor m_b;
};
TORCH_MODULE(LocallyConnected2D);


static int compute_output_size(int w, int f, int s, int p) {
    return (w-f+2*p)/s + 1;
}

struct DQN : torch::nn::Module {
    DQN(const int64_t _in_channels,
        const int64_t _input_size, 
        const int64_t _output_size)
      : 
          local1(register_module("local1", LocallyConnected2D(_in_channels, 4, compute_output_size(_input_size, 3, 1, 1), compute_output_size(_input_size, 3, 1, 1), 1, 1))),
          conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 3).stride(1).padding(1).bias(true)))),
          bn_1(register_module("bn_1", torch::nn::BatchNorm2d(8))),
          conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).stride(1).padding(1).bias(true)))),
          bn_2(register_module("bn_2", torch::nn::BatchNorm2d(16))),
          conv3(register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1).bias(true)))),          
          bn_3(register_module("bn_3", torch::nn::BatchNorm2d(32))),
          conv4(register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1).bias(true)))),          
          bn_4(register_module("bn_4", torch::nn::BatchNorm2d(64))),
          conv5(register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(true)))),          
          bn_5(register_module("bn_5", torch::nn::BatchNorm2d(128))),
          conv6(register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(true)))),          
          bn_6(register_module("bn_6", torch::nn::BatchNorm2d(256))),
          conv7(register_module("conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1).bias(true)))),          
          bn_7(register_module("bn_7", torch::nn::BatchNorm2d(256))),
          linear1_v(register_module("linear1_v", torch::nn::Linear(torch::nn::LinearOptions(61954, 32).bias(true)))),
          linear2_v(register_module("linear2_v", torch::nn::Linear(torch::nn::LinearOptions(32, 1).bias(true)))),
          linear1_a(register_module("linear1_a", torch::nn::Linear(torch::nn::LinearOptions(61954, 32).bias(true)))),
          linear2_a(register_module("linear2_a", torch::nn::Linear(torch::nn::LinearOptions(32, _output_size).bias(true)))) {}

    torch::Tensor forward(torch::Tensor geometric, torch::Tensor time_series) {
        auto input = torch::sigmoid(local1(geometric));
        input = torch::relu(bn_1(conv1(input)));
        input = torch::relu(bn_2(conv2(input)));
        input = torch::relu(bn_3(conv3(input)));
        input = torch::relu(bn_4(conv4(input)));
        input = torch::relu(bn_5(conv5(input)));
        input = torch::relu(bn_6(conv6(input)));
        input = torch::relu(bn_7(conv7(input)));
        input = input.view({-1,input.size(1)*input.size(2)*input.size(3)});
        input = torch::cat({input, time_series}, 1);
        auto advantage = torch::sigmoid(linear1_a(input));
        advantage = torch::sigmoid(linear2_a(advantage));

        auto value = torch::sigmoid(linear1_v(input));
        value = torch::sigmoid(linear2_v(value));

        return value + advantage - advantage.mean();
    }

    LocallyConnected2D local1;
    torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5, conv6, conv7;
    torch::nn::BatchNorm2d bn_1, bn_2, bn_3, bn_4, bn_5, bn_6, bn_7;
    torch::nn::Linear linear1_v,linear2_v,linear1_a,linear2_a;
};



#endif /* DQN_HPP */
