/* hog_run: benchmark a trained HOG model. Part of the HOG Trainer suite.
 *
 * Copyright (c) 2015 University of Nevada, Las Vegas
 */

#include <stdio.h>
#include <sstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../common/ht_common.hpp"
#include "../common/ht_image_paths.hpp"

using namespace cv;
using namespace std;

const option::Descriptor usage[] =
{
  {UNKNOWN, 0, "", "", Arg::Unknown, "USAGE: hog_run [options] svm_file\n\n"
                                     "Options:" },
  {HELP, 0, "", "help", Arg::None, "  --help  \t\tPrint this text." },
  {POS_PATH, 0, "p", "pos", Arg::Path, "  --pos <path>, \t-p <path>  \tSpecifies the positive test images path."},
  {NEG_PATH, 0, "n", "neg", Arg::Path, "  --neg <path>, \t-n <path>  \tSpecifies the negative test images path."},
  {SIZE_X, 0, "x", "", Arg::Numeric, "  -x <n>  \t\tSpecifies an X height for the test images in pixels (default: 64)."},
  {SIZE_Y, 0, "y", "", Arg::Numeric, "  -y <n>  \t\tSpecifies a Y height for the test images in pixels (default: 128)."},
  {0, 0, 0, 0, 0, 0}
};

string svm_type_as_string(int svm_type) {
  switch(svm_type) {
    case CvSVM::C_SVC:
    return string("C-Support Vector Classification");
    case CvSVM::NU_SVC:
    return string("Nu-Support Vector Classification");
    case CvSVM::ONE_CLASS:
    return string("Distribution Estimation (One-class SVM)");
    case CvSVM::EPS_SVR:
    return string("Epsilon-Support Vector Regression");
    case CvSVM::NU_SVR:
    return string("Nu-Support Vector Regression");
    default:
    return string("Unknown SVM Type");
  }
}

string svm_kernel_as_string(int svm_kernel) {
  switch(svm_kernel) {
    case CvSVM::LINEAR:
    return string("Linear");
    case CvSVM::POLY:
    return string("Polynomial");
    case CvSVM::RBF:
    return string("Radial Basis Function");
    case CvSVM::SIGMOID:
    return string("Sigmoid");
    default:
    return string("Unknown SVM Kernel");
  }
}

unsigned int process_images(vector<string>& imagePaths,
                        unsigned int size_x, unsigned int size_y, CvSVM &svm,
                        bool positive) {
  unsigned int row = 0;
  unsigned int misclassified = 0;
  auto totalPaths = imagePaths.size();

  saveCursor();
  for(auto path : imagePaths) {
    restoreCursor();
    if(positive) {
      progress(row, totalPaths, "Testing against positive images...");
    }
    else {
      progress(row, totalPaths, "Testing against negative images...");
    }

    // Load the image and convert it to grayscale in one step:
    auto image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    resize(image, image, Size(size_x, size_y));

    HOGDescriptor hog(Size(size_x, size_y), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    vector<float> v;
    vector<Point> l;
    hog.compute(image, v, Size(0,0), Size(0,0), l);

    Mat fm = Mat(v);
    int result = svm.predict(fm);
    // Assume we're using a classification (not regression) model; thus
    // with returnDLVal = false, 1 is a positive label and -1 is a negative label.
    if(positive && result == -1) {
      misclassified = misclassified + 1;
    }
    else if(!positive && result == 1) {
      misclassified = misclassified + 1;
    }

    row = row + 1;
    // 'image' should be released here as it falls out of scope...
  }
  fprintf(stderr, " Done.\n");

  return misclassified;
}

int main(int argc, char* argv[]) {
  argc -= (argc>0); argv += (argc>0); // Skip argv[0] if present.
  option::Stats stats(usage, argc, argv);
  unique_ptr<option::Option> options(new option::Option[stats.options_max]);
  unique_ptr<option::Option> buffer(new option::Option[stats.buffer_max]);
  option::Parser parse(usage, argc, argv, options.get(), buffer.get());

  string pos_dir = "pos";
  string neg_dir = "neg";
  unsigned int image_x = 64;
  unsigned int image_y = 128;

  if(parse.error()) {
    return 1;
  }

  if(options.get()[HELP] || parse.nonOptionsCount() != 1) {
    int columns = getenv("COLUMNS") ? atoi(getenv("COLUMNS")) : 80;
    option::printUsage(fwrite, stdout, usage, columns);
    return 0;
  }

  if(options.get()[POS_PATH]) {
    pos_dir = options.get()[POS_PATH].last()->arg;
  }

  if(options.get()[NEG_PATH]) {
    neg_dir = options.get()[NEG_PATH].last()->arg;
  }

  if(options.get()[SIZE_X]) {
    string x_str = options.get()[SIZE_X].last()->arg;
    istringstream(x_str) >> image_x;
  }

  if(options.get()[SIZE_Y]) {
    string y_str = options.get()[SIZE_Y].last()->arg;
    istringstream(y_str) >> image_y;
  }

  string svm_path = parse.nonOption(0);

  CvSVM svm;
  svm.load(svm_path.c_str());
  auto params = svm.get_params();
  auto svm_type = svm_type_as_string(params.svm_type);
  auto svm_kernel = svm_kernel_as_string(params.kernel_type);
  double svm_c = params.C;
  double svm_gamma = params.gamma;
  double svm_nu = params.nu;
  double svm_coef0 = params.coef0;
  double svm_degree = params.degree;
  printf("Using SVM model: '%s'.\n", svm_path.c_str());
  printf("Type: %s\n", svm_type.c_str());
  printf("Kernel: %s\n", svm_kernel.c_str());
  printf("Parameters: C=%.3f; gamma=%.3f; nu=%.3f; coef0=%.3f; degree=%.3f\n",
        svm_c, svm_gamma, svm_nu, svm_coef0, svm_degree);
  printf("\n");

  vector<string> imagePaths;
  fprintf(stderr, "Using positive test image directory '%s'...\n", pos_dir.c_str());
  if(!get_image_paths_into(pos_dir, imagePaths)) {
    fprintf(stderr, "Couldn't open positive test image directory '%s'.\n", pos_dir.c_str());
    return 1;
  }
  auto num_pos = imagePaths.size();
  fprintf(stderr, "Found %zu positive test images.\n", num_pos);
  unsigned int wrong_pos = process_images(imagePaths, image_x, image_y, svm, true);
  printf("Misclassified %u of %zu positive images (%.2f%% accuracy).\n", wrong_pos, num_pos, ((float)num_pos - (float)wrong_pos) / (float)num_pos * 100.0);
  fprintf(stderr, "\n");

  imagePaths.clear();
  fprintf(stderr, "Using negative test image directory '%s'...\n", neg_dir.c_str());
  if(!get_image_paths_into(neg_dir, imagePaths)) {
    fprintf(stderr, "Couldn't open negative test image directory '%s'\n", neg_dir.c_str());
    return 1;
  }
  auto num_neg = imagePaths.size();
  fprintf(stderr, "Found %zu negative test images.\n", num_neg);
  unsigned int wrong_neg = process_images(imagePaths, image_x, image_y, svm, false);
  printf("Misclassified %u of %zu negative images (%.2f%% accuracy).\n", wrong_neg, num_neg, ((float)num_neg - (float)wrong_neg) / (float)num_neg * 100.0);


  return 0;
}
