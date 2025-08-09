#include "neural_network.hpp"

inline void normalizeInto(Eigen::MatrixXf& mat, const std::vector<std::vector<uint8_t>>& input) {
    const int rows = mat.rows();
    const int cols = mat.cols();

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            mat(row, col) = static_cast<float>(input[col][row]) / 255.0f;
        }
    }
}

inline MatrixXf oneHotEncoding(vector<uint8_t> label, int outputAmount, const int batchSize) {
    MatrixXf newMat = Eigen::MatrixXf::Zero(outputAmount, batchSize);

    for (int i = 0; i < batchSize; i++) {
        newMat(label[i],i) = 1.0f;
    }
    return newMat;
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
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    // Constants
    const int inputLength = dataset.training_images[0].size();
    const int outputLength = 10;
    const float learningRate = 0.0095f;
    const int batchSize = 10;
    const int epochs = 5;

    // Mutable variables
    MatrixXf inputs;
    inputs.resize(784, batchSize);
    std::vector<std::vector<uint8_t>> rawInputs;
    std::vector<uint8_t> labels(batchSize);

    // Network Setup
    NN::Network nn(batchSize);
    nn.addLayer(32, LEAKY_RELU);
    nn.addLayer(32, LEAKY_RELU);
    nn.addLayer(outputLength, LEAKY_RELU);
    nn.setup(inputLength);

    for (int e = 0; e < epochs; e++) {
        int numBatches = dataset.training_images.size() / batchSize;
        float epochLoss = 0.0f;
        int correctPredictions = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < numBatches; b++) {
            rawInputs.clear();

            for (int i = 0; i < batchSize; i++) {
                int index = b * batchSize + i;
                rawInputs.push_back(dataset.training_images[index]);
                labels[i] = dataset.training_labels[index];
            }

            normalizeInto(inputs, rawInputs);
            MatrixXf targets = oneHotEncoding(labels, outputLength, batchSize);

            // Forwardpass
            nn.forwardPass(inputs);

            // Get output from last layer
            const MatrixXf& predictions = nn.layers.back()->activations;

            // Compute batch loss
            epochLoss += lossMSE(targets, predictions);

            // Compute accuracy
            for (int i = 0; i < batchSize; i++) {
                VectorXf col = predictions.col(i);
                int predicted = argmax(col);
                if (predicted == labels[i]) {
                    correctPredictions++;
                }
            }

            // Backpropagation
            nn.backwardPass(learningRate, targets, inputs);
        }

        float avgLoss = epochLoss / numBatches;
        float accuracy = static_cast<float>(correctPredictions) / (numBatches * batchSize);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Epoch " << (e + 1) << "/" << epochs
                << " | Loss: " << avgLoss
                << " | Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%"
                << " | Time taken: " << duration.count() << " seconds" << "" << std::endl;
    }

    // Test the trained nn
    int correctTestPredictions = 0;
    const int testSize = dataset.test_images.size();
    MatrixXf testInput(784, 1);

    for (int i = 0; i < testSize; ++i) {
        for (int j = 0; j < 784; ++j) {
            testInput(j, 0) = static_cast<float>(dataset.test_images[i][j]) / 255.0f;
        }

        nn.forwardPass(testInput);
        const MatrixXf& output = nn.layers.back()->activations;
        int predicted = argmax(output.col(0));
        int actual = dataset.test_labels[i];

        if (predicted == actual) {
            correctTestPredictions++;
        }
    }

    float testAccuracy = static_cast<float>(correctTestPredictions) / testSize;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2)
              << testAccuracy * 100.0f << "%" << std::endl;
}

int main() {
    Eigen::setNbThreads(12);
    std::cout << "Program started\n";
    train();
    std::cout << "Training finished\n";
    return 0;
}
