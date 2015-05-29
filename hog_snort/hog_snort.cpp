#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../common/ht_common.hpp"

using namespace cv;
using namespace std;

const option::Descriptor usage[] =
{
  {UNKNOWN, 0, "", "", Arg::Unknown, "USAGE: hog_snort [options] feature_file\n\n"
                                     "Options:" },
  {HELP, 0, "", "help", Arg::None, "  --help  \tPrint this text." },
  {POS_PATH, 0, "p", "path", Arg::Path, "  --path <path>, \t-p <path>  \tSpecifies the path for the image examples."},
  {0, 0, 0, 0, 0, 0}
};

bool is_valid_file_extension(string ext) {
  if(strcmp("bmp", ext.c_str()) == 0) {
    return true;
  }
  else if(strcmp("jpg", ext.c_str()) == 0) {
    return true;
  }
  else if(strcmp("jpeg", ext.c_str()) == 0) {
    return true;
  }
  else if(strcmp("png", ext.c_str()) == 0) {
    return true;
  }
  else if(strcmp("ppm", ext.c_str()) == 0) {
    return true;
  }
  else if(strcmp("pgm", ext.c_str()) == 0) {
    return true;
  }
  return false;
}

bool get_image_paths_into(string dir, vector<string>& into) {
  auto dirp = opendir(dir.c_str());
  if(dirp != NULL) {
    dirent *dp;
    while((dp = readdir(dirp))) {
      auto filename = string(dp->d_name);
      auto ext_pos = string(dp->d_name).find_last_of('.');
      auto ext = filename.substr(ext_pos + 1);
      // Convert extension to lowercase:
      transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

      if(dp->d_type & DT_DIR) {
        continue; // Skip directories
      }
      else if(!is_valid_file_extension(ext)) {
        fprintf(stderr, "Skipping %s...\n", dp->d_name);
        continue;
      }
      string imagePath = string(dir);
      imagePath += "/";
      imagePath += filename;
      into.push_back(imagePath);
    }
    closedir(dirp);
    return true;
  }
  else {
    return false;
  }
}

void write_headers(bool valid, int length, int width, ofstream &f) {
  f.seekp(0);

  if(valid) {
    f.write("HOGSNRT", 7);
  }
  else {
    f.write("INVALID", 7);
  }
  f.write((char *)&length, sizeof(int));
  f.write((char *)&width, sizeof(int));
}

void process_images(vector<string>& imagePaths, ofstream &featureFile) {
  unsigned int row = 0;
  unsigned int column = 0;

  auto totalPaths = imagePaths.size();
  fprintf(stderr, "Found %zu examples.\n", totalPaths);

  // Preallocate space in the file for the headers:
  write_headers(false, 0, 0, featureFile);

  saveCursor();
  for(auto path : imagePaths) {
    restoreCursor();
    progress(row, totalPaths, "Processing examples...");

    // Load the image and convert it to grayscale in one step:
    auto image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    resize(image, image, Size(64, 128));

    HOGDescriptor hog(Size(64, 128), Size(8, 8), Size(4, 4), Size(4, 4), 9);
    vector<float> v;
    vector<Point> l;
    hog.compute(image, v, Size(0,0), Size(0,0), l);
    if(column == 0) {
      column = v.size();
      //features.create(totalPaths, column, CV_32FC1);
    }
    featureFile.write((char *)v.data(), sizeof(float) * v.size());
    //features.push_back(f);
    //memcpy(&(features.data[column * row * sizeof(float)]),
    //        v.data(), column * sizeof(float));
    //values.push_back(v);
    //locations.push_back(l);

    row = row + 1;
    // 'image' should be released here as it falls out of scope...
  }

  write_headers(true, row, column, featureFile);
  fprintf(stderr, " Done.\n");
}

int main(int argc, char* argv[]) {
  argc -= (argc>0); argv += (argc>0); // Skip argv[0] if present.
  option::Stats stats(usage, argc, argv);
  unique_ptr<option::Option> options(new option::Option[stats.options_max]);
  unique_ptr<option::Option> buffer(new option::Option[stats.buffer_max]);
  option::Parser parse(usage, argc, argv, options.get(), buffer.get());

  string pos_dir = "pos";

  if(parse.error()) {
    return 1;
  }

  if(options.get()[HELP] || parse.nonOptionsCount() != 1) {
    int columns = getenv("COLUMNS") ? atoi(getenv("COLUMNS")) : 80;
    option::printUsage(fwrite, stdout, usage, columns);
    return 0;
  }

  string feature_path = parse.nonOption(0);

  if(options.get()[POS_PATH]) {
    pos_dir = options.get()[POS_PATH].last()->arg;
  }

  vector<string> imagePaths;
  fprintf(stderr, "Using image directory '%s'...\n", pos_dir.c_str());
  if(!get_image_paths_into(pos_dir, imagePaths)) {
    fprintf(stderr, "Couldn't open image directory '%s'\n", pos_dir.c_str());
    return false;
  }

  //Mat M;
  //vector<vector<Point>> feature_locations;
  //FileStorage featureFile(feature_path, FileStorage::WRITE);
  ofstream featureFile(feature_path, ofstream::binary);
  process_images(imagePaths, featureFile);

  printf("Wrote features to '%s'.\n", feature_path.c_str());
  //write(featureFile, "Descriptor_of_images", M);

  //M.release();
  //featureFile.release();
  featureFile.close();
  return 0;
}
