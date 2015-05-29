#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "../common/ht_common.hpp"

using namespace cv;
using namespace std;

const option::Descriptor usage[] =
{
  {UNKNOWN, 0, "", "", Arg::Unknown, "USAGE: hog_trainer [options] svm_file\n\n"
                                     "Options:" },
  {HELP, 0, "", "help", Arg::None, "  --help  \tPrint this text." },
  {POS_PATH, 0, "p", "pos", Arg::Path, "  --pos <path>, \t-p <path>  \tSpecifies the positive feature file."},
  {NEG_PATH, 0, "n", "neg", Arg::Path, "  --neg <path>, \t-n <path>  \tSpecifies the negative feature file."},
  {0, 0, 0, 0, 0, 0}
};

bool open_features(unsigned int &length, unsigned int &width, ifstream &f, const char *label) {
  char header[8];
  f.read(header, 7);
  header[7] = '\0';
  if(strcmp("HOGSNRT", header) != 0) {
    fprintf(stderr, "Invalid %s features file header\n", label);
    return false;
  }

  f.read((char *)&length, sizeof(int));
  f.read((char *)&width, sizeof(int));
  fprintf(stderr, "Found %d %s examples...\n", length, label);

  return true;
}

bool read_features_into(unsigned int length, unsigned int width, unsigned int start, ifstream &f, Mat &features, const char *label)  {
  saveCursor();
  for(unsigned int r = 0; r < length; ++r) {
    restoreCursor();
    char progressMessage[255];
    snprintf(progressMessage, 255, "Reading %s examples...", label);
    progress(r, length, progressMessage);
    for(unsigned int c = 0; c < width; ++c) {
      float val;
      f.read((char *)&val, sizeof(float));
      if(f.bad()) {
        fprintf(stderr, "Prematurely truncated %s examples file.\n", label);
        return false;
      }
      features.row(r + start).col(c) = val;
    }
  }
  printf(" Done.\n");

  return true;
}

int main(int argc, char* argv[]) {
  argc -= (argc>0); argv += (argc>0); // Skip argv[0] if present.
  option::Stats stats(usage, argc, argv);
  unique_ptr<option::Option> options(new option::Option[stats.options_max]);
  unique_ptr<option::Option> buffer(new option::Option[stats.buffer_max]);
  option::Parser parse(usage, argc, argv, options.get(), buffer.get());

  string pos_path = "positive.bin";
  string neg_path = "negative.bin";

  if(parse.error()) {
    return 1;
  }

  if(options.get()[HELP] || parse.nonOptionsCount() != 1) {
    int columns = getenv("COLUMNS") ? atoi(getenv("COLUMNS")) : 80;
    option::printUsage(fwrite, stdout, usage, columns);
    return 0;
  }

  string svm_path = parse.nonOption(0);

  if(options.get()[POS_PATH]) {
    pos_path = options.get()[POS_PATH].last()->arg;
  }

  if(options.get()[NEG_PATH]) {
    neg_path = options.get()[NEG_PATH].last()->arg;
  }

  //FileStorage positiveXml(pos_path, FileStorage::READ);
  ifstream positiveFile(pos_path, ifstream::binary);
  if(positiveFile.good()) {
    fprintf(stderr, "Using positive features file '%s'...\n", pos_path.c_str());
  }
  else {
    fprintf(stderr, "Couldn't open positive features file '%s'.\n", pos_path.c_str());
    return 1;
  }

  unsigned int p_length;
  unsigned int width;
  if(!open_features(p_length, width, positiveFile, "positive")) {
    return 1;
  }

  Mat features(p_length, width, CV_32FC1);
  read_features_into(p_length, width, 0, positiveFile, features, "positive");
  positiveFile.close();

  ifstream negativeFile(neg_path, ifstream::binary);
  if(negativeFile.good()) {
    fprintf(stderr, "Using negative features file '%s'...\n", neg_path.c_str());
  }
  else {
    fprintf(stderr, "Couldn't open negative features file '%s'.\n", neg_path.c_str());
    return 1;
  }

  unsigned int n_length;
  if(!open_features(n_length, width, negativeFile, "negative")) {
    return 1;
  }

  features.resize(p_length + n_length);
  read_features_into(n_length, width, p_length, negativeFile, features, "negative");
  negativeFile.close();

  Mat labels(p_length + n_length, 1, CV_32FC1, Scalar(-1.0));
  labels.rowRange(0, p_length) = Scalar(1.0);

  fprintf(stderr, "Training the HOG...");
  CvSVM svm;
  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
  svm.train(features, labels, Mat(), Mat(), params);
  //svm.train_auto(features, labels, Mat(), Mat(), params);
  fprintf(stderr, " Done.\n");

  fprintf(stderr, "Writing trained model to '%s'...\n", svm_path.c_str());
  svm.save(svm_path.c_str());

  labels.release();
  features.release();

  return 0;
}
