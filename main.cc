#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <experimental/filesystem>
#include <iostream>
#include <regex>
#include <sstream>

namespace fs = std::experimental::filesystem;
using namespace dlib;

// Principal Component Analysis
std::vector<matrix<double>> PCA(const std::vector<matrix<double>>& data, float& epsilon)
{
    vector_normalizer_pca<matrix<double>> pca;
    pca.train(data, epsilon);
    std::vector<matrix<double>> new_data;
    new_data.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        new_data.emplace_back(pca(data[i]));
    }
    return new_data;
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        if (fs::exists(argv[1])) {

            // Reading the iris dataset
            std::stringstream str_stream;
            {
                // replace categorial values with numeric ones
                std::ifstream data_stream(argv[1]);
                std::string data_string((std::istreambuf_iterator<char>(data_stream)),
                    std::istreambuf_iterator<char>());

                //  replace string labels
                data_string = std::regex_replace(data_string, std::regex("Iris-setosa"), "1");
                data_string = std::regex_replace(data_string, std::regex("Iris-versicolor"), "2");
                data_string = std::regex_replace(data_string, std::regex("Iris-virginica"), "3");

                str_stream << data_string;
            }
            matrix<double> data;
            str_stream >> data;

            size_t n = 150;
            matrix<double> samples = subm(data, range(0, data.nr() - 1), range(0, 3));
            matrix<double> targets = subm(data, range(0, data.nr() - 1), range(4, 4));

            std::vector<matrix<double>> x;
            std::vector<matrix<double>> y_data;

            for (size_t r = 0; r < n; ++r) {
                x.push_back(rowm(samples, r));
                y_data.push_back(rowm(targets, r));
            }

            // Shuffle the dataset
            randomize_samples(x, y_data);
            std::vector<float> y (y_data.cbegin(), y_data.cend());

            float epsilon = 0;
            std::cout << "Epsilon is a number that controls how lossy the pca transform will be\n";
            std::cout << "Large value of epsilon results in a bigger number of features\n";
            std::cout << "Epsilon must be between 0 and 1.\n";
            std::cout << "Please refer to statistics_abstract.h from dlib's website for reference.\n";
            std::cout << "For iris dataset we can reduce the number of features to 3,2, and 1.\n";
            std::cout << "The epsilon values for those reductions are 0.99,0.8,and 0.7 respectively\n";
            std::cout << "Please choose the epsilon value (must be between 0 and 1): ";

            do {
                std::cin >> epsilon;
                if (epsilon < 0 || epsilon > 1) {
                    std::cout << "Input must be between 0 and 1" << std::endl;
                    std::cout << "Please try again: ";
                }
            } while (epsilon < 0 || epsilon > 1);

            // Testing set
            int test_num = 50;
            std::vector<matrix<double>> test_data;
            std::vector<float> test_targets;
            for (int row = 0; row < test_num; ++row) {
                test_data.emplace_back(reshape_to_column_vector(x[row]));
                test_targets.emplace_back(y[row]);
            }

            // PCA application on testing set
            auto pca_test = PCA(test_data, epsilon);

        
            // Training set
            std::vector<matrix<double>> train_data;
            std::vector<float> train_targets;
            for (int row = test_num; row < samples.nr(); ++row) {
                train_data.emplace_back(reshape_to_column_vector(x[row]));
                train_targets.emplace_back(y[row]);
            }

            // PCA application on training set
            auto pca_train = PCA(train_data, epsilon);

            // MLP network
            using NetworkType = loss_mean_squared<
                fc<1, htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<double>>>>>>>>>>;

            NetworkType mlp;

            float weight_decay = 0.0001f;
            float momentum = 0.9f;
            sgd solver(weight_decay, momentum);
            dnn_trainer<NetworkType> trainer(mlp, solver);
            trainer.set_learning_rate(0.001);
            trainer.set_learning_rate_shrink_factor(1);
            trainer.set_max_num_epochs(1000);
            trainer.be_verbose();
            trainer.train(pca_train, train_targets);
            mlp.clean();

            auto predicted_labels = mlp(pca_test);
            int num_right = 0;
            int num_wrong = 0;

            for (size_t i = 0; i < test_targets.size(); ++i) {
                if (round(predicted_labels[i]) == test_targets[i])
                    ++num_right;

                else
                    ++num_wrong;
            }

            std::cout << "PCA with " << test_data[0].size() - pca_test[0].size() << " feature(s) reduction." << std::endl;
            std::cout << "testing num_right: " << num_right << std::endl;
            std::cout << "testing num_wrong: " << num_wrong << std::endl;
            std::cout << "testing accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) * 100 << std::endl;

        } else {
            std::cerr << "Invalid file path " << argv[1] << std::endl;
        }
    } else {
        std::cerr << "Please provide a path to a dataset\n";
    }

    return 0;
}
