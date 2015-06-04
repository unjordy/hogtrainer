#ifndef HT_COMMON_HPP
#define HT_COMMON_HPP

#include <stdio.h>
#include "optionparser.h"

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

enum optionIndex {UNKNOWN, HELP, POS_PATH, NEG_PATH, AUTO_TRAIN};

static void saveCursor(void) {
  fwrite("\033[s", sizeof(char), 3, stderr);
}

static void restoreCursor(void) {
  fwrite("\033[u", sizeof(char), 3, stderr);
}

static void progress(unsigned int current, unsigned int total, const char *message) {
  static int lastProgress = 100;
  int progress = (current + 1) * 100 / total;

  if(progress != lastProgress) {
    lastProgress = progress;
    fprintf(stderr, "[%3d%%] %s", progress, message);
    fflush(stderr);
  }
}

#endif /* HT_COMMON_HPP */
