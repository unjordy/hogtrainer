#ifndef HT_IMAGE_PATHS_HPP
#define HT_IMAGE_PATHS_HPP

#include <stdio.h>
#include <dirent.h>
#include <string>
#include <vector>

static bool is_valid_file_extension(std::string ext) {
  auto e = ext.c_str();
  if(strcmp("bmp", e) == 0) {
    return true;
  }
  else if(strcmp("jpg", e) == 0) {
    return true;
  }
  else if(strcmp("jpeg", e) == 0) {
    return true;
  }
  else if(strcmp("png", e) == 0) {
    return true;
  }
  else if(strcmp("ppm", e) == 0) {
    return true;
  }
  else if(strcmp("pgm", e) == 0) {
    return true;
  }
  return false;
}

static bool get_image_paths_into(std::string dir, std::vector<std::string>& into) {
  auto dirp = opendir(dir.c_str());
  if(dirp != NULL) {
    dirent *dp;
    while((dp = readdir(dirp))) {
      auto filename = std::string(dp->d_name);
      auto ext_pos = std::string(dp->d_name).find_last_of('.');
      auto ext = filename.substr(ext_pos + 1);
      // Convert extension to lowercase:
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

      if(dp->d_type & DT_DIR) {
        continue; // Skip directories
      }
      else if(!is_valid_file_extension(ext)) {
        fprintf(stderr, "Skipping %s...\n", dp->d_name);
        continue;
      }
      auto imagePath = std::string(dir);
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

#endif /* HT_IMAGE_PATHS_HPP */
