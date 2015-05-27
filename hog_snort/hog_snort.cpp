#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../common/optionparser.h"

using namespace cv;
using namespace std;

struct Arg: public option::Arg {
  static void printError(const char* msg1, const option::Option& opt, const char* msg2) {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

  static option::ArgStatus Unknown(const option::Option& opt, bool msg) {
    if(msg) {
      printError("Unknown option '", opt, "'\n");
    }
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Path(const option::Option& opt, bool msg) {
    if(opt.arg != 0) {
      return option::ARG_OK;
    }

    if(msg) {
      printError("Option '", opt, "' requires an argument\n");
    }
    return option::ARG_ILLEGAL;
  }
};

enum optionIndex {UNKNOWN, HELP, POS_PATH, NEG_PATH};
const option::Descriptor usage[] =
{
  {UNKNOWN, 0, "", "", Arg::Unknown, "USAGE: hog_snort [options] xml_file\n\n"
                                     "Options:" },
  {HELP, 0, "", "help", Arg::None, "  --help  \tPrint this text." },
  {POS_PATH, 0, "p", "path", Arg::Path, "  --path <path>, \t-p <path>  \tSpecifies the path for the image examples."},
  {0, 0, 0, 0, 0, 0}
};

static void saveCursor(void) {
  fwrite("\033[s", sizeof(char), 3, stderr);
}

static void restoreCursor(void) {
  fwrite("\033[u", sizeof(char), 3, stderr);
}

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

void read_images(vector<string>& imagePaths, Mat& features) {
  unsigned int row = 0;
  int lastProgress = 100;
  int column = 0;

  auto totalPaths = imagePaths.size();
  fprintf(stderr, "Found %zu examples.\n", totalPaths);

  saveCursor();
  for(auto path : imagePaths) {
    restoreCursor();
    int progress = (row + 1) * 100 / totalPaths;
    if(progress != lastProgress) {
      lastProgress = progress;
      fprintf(stderr, "[%3d%%] Processing examples...", progress);
      fflush(stderr);
    }

    // Load the image and convert it to grayscale in one step:
    auto image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    resize(image, image, Size(64, 128));

    HOGDescriptor hog(Size(64, 128), Size(8, 8), Size(4, 4), Size(4, 4), 9);
    vector<float> v;
    vector<Point> l;
    hog.compute(image, v, Size(0,0), Size(0,0), l);
    if(column == 0) {
      column = v.size();
      features.create(totalPaths, column, CV_32F);
    }
    memcpy(&(features.data[column * row * sizeof(float)]),
            v.data(), column * sizeof(float));
    //values.push_back(v);
    //locations.push_back(l);

    row = row + 1;
    // 'image' should be released here as it falls out of scope...
  }
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

  string xml_path = parse.nonOption(0);

  if(options.get()[POS_PATH]) {
    pos_dir = options.get()[POS_PATH].last()->arg;
  }

  vector<string> imagePaths;
  fprintf(stderr, "Using image directory '%s'...\n", pos_dir.c_str());
  if(!get_image_paths_into(pos_dir, imagePaths)) {
    fprintf(stderr, "Couldn't open image directory '%s'\n", pos_dir.c_str());
    return false;
  }

  Mat M;
  //vector<vector<Point>> feature_locations;
  read_images(imagePaths, M);

  printf("Writing features to '%s'.\n", xml_path.c_str());
  FileStorage featureXml(xml_path, FileStorage::WRITE);
  write(featureXml, "Descriptor_of_images", M);

  M.release();
  featureXml.release();
  return 0;
}
