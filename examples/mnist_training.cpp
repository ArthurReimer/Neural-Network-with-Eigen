// g++ -O3 -fopenmp mnist_training.cpp -I../src -I../external/eigen -I../external/mnist/include -std=c++17 -o mnist_training.exe -lgomp
// g++ -Og mnist_training.cpp -I../src -I../external/eigen -I../external/mnist/include -std=c++17 -o mnist_training.exe

#include "neural_network.hpp"



class Timer {
    public:
        std::chrono::steady_clock::time_point startTime;

    void startTimer() {
        startTime = std::chrono::steady_clock::now();
    }

    void endTimerMilliseconds() {
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() << "ms\n";
    }

    void endTimerSeconds() {
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count() << "sec\n";
    }
};


using VectorXu8 = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;

inline void normalizeInto(Eigen::VectorXf& vec, const std::vector<uint8_t>& input) {
    vec = Eigen::Map<const VectorXu8>(input.data(), input.size()).cast<float>() / 255.0f;
}

inline MatrixXf oneHotEncoding(uint8_t label, int outputAmount) {
    VectorXf vec = VectorXf::Zero(outputAmount);
    vec(label) = 1;
    return vec;
}

inline int argmax(const VectorXf& vec) {
    Eigen::Index maxIndex;
    vec.maxCoeff(&maxIndex);
    return static_cast<int>(maxIndex);
}

inline float lossMSE(const MatrixXf& expected, const MatrixXf& output) {
    return ((expected - output).array().square().sum()) / expected.cols();
}

void train() {
    // MNIST dataset
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../external/mnist");

    // Constants
    const int inputLength = dataset.training_images[0].size();
    const int outputLength = 10;
    const int epochs = 400;

    double learningRate = 0.0005;

    // Network Setup
    NeuralNetwork nn(inputLength);
    nn.addLayer(256, LEAKY_RELU);
    nn.addLayer(128, LEAKY_RELU);
    nn.addLayer(outputLength, SIGMOID);

    VectorXf inputs(inputLength);
    
    Timer timer = Timer();
    timer.startTimer();

    std::cout << "Neural Network Model" << "\n"
            << "  Denselayer amount - " << nn.layers.size() << "\n"
            << "  Weight amount - " << nn.getWeightAmount() << "\n"
            << "  Bias amount - " << nn.getBiasAmount() << "\n\n";

    std::cout << "Starting Training" << "\n";

    // training

    for (int e = 0; e < epochs; e++) {
        int correct = 0;
        for (int i = 0; i < dataset.training_images.size(); i++) {
            
            normalizeInto(inputs, dataset.training_images[i]);
            nn.forwardpass(inputs);

            int label = static_cast<int>(dataset.training_labels[i]);
            int prediction = argmax(nn.layers[nn.layers.size()-1].activations);
            if (prediction == label) {
                correct += 1;
            }


            VectorXf target = oneHotEncoding(label, outputLength);
            nn.backpropagation(inputs, learningRate, target);
        }
        float accuracy = static_cast<float>(correct) / dataset.training_images.size();
        std::cout << "Epoch " << e+1 <<" Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "% ";
        timer.endTimerSeconds();
    }

    // testing

    timer.startTimer();
    VectorXf inputsTestData(inputLength);
    int correct = 0;

    for (int i = 0; i < dataset.test_images.size(); i++) {
        
        normalizeInto(inputsTestData, dataset.test_images[i]);
        nn.forwardpass(inputsTestData);

        int label = static_cast<int>(dataset.test_labels[i]);
        int prediction = argmax(nn.layers[nn.layers.size()-1].activations);
        if (prediction == label) {
            correct += 1;
        }
    }

    std::cout << std::endl << "Testing trained model" << std::endl;
    float accuracy = static_cast<float>(correct) / dataset.test_images.size();
    std::cout << "Testdata Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "% ";
    timer.endTimerSeconds();
}

int main() {
    train();
    return 0;
}
