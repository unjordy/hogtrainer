The HOG Trainer
===============

The HOG Trainer is a suite of memory-efficient utilities used to train [HOG (Histogram of Oriented Gradients)][1] models for computational image recognition and feature extraction. This suite makes extensive use of OpenCV and its SVM functionality, and is under the same license (BSD-3-Clause).

###Dependencies
For building, the HOG Trainer suite depends on the [`tup` build system](http://gittup.org/tup/) and `pkg-config`. Its runtime dependencies are OpenCV 2.x (developed with 2.11), `libjpeg`, `libpng`, `libtiff`, and the Boost C++ libraries (we use version 1.57). All of these should be available from your operating system's package manager. **If you are using OpenCV with CUDA support**: set the environment variable `$CUDA_PATH` to point to the directory where your CUDA SDK is installed.

###Building
Run this anywhere in the HOG Trainer source tree:
```
tup
```
The compiled binaries should end up in the `bin` subdirectory. The build process doesn't yet support compiling just one utility, and it doesn't include any build variants. You will be able to set some build options by modifying the top-level `Tuprules.tup` file.

###Summary of the included programs
The HOG Trainer currently consists of three utilities:
* `hog_snort` ingests a directory of images and outputs a binary feature file
* `hog_trainer` takes a positive and a negative feature file and produces an OpenCV-compatible SVM model in XML format
* `hog_run` benchmarks a trained XML model against a directory of positive examples and a directory of negative examples

###Example of typical usage
Assume that the HOG Trainer utilities are in your `$PATH` and that your current directory has a subdirectory `person_set` with this structure:
```
person_set/training/pos   (a set of positive training images)
person_set/training/neg   (a set of negative training images)
person_set/testing/pos    (a set of positive testing images)
person_set/testing/neg    (a set of negative testing images)
```
The first step is to ingest your positive and negative training sets with `hog_snort`. This is accomplished with the following commands:
```
hog_snort --path person_set/training/pos positive.bin
hog_snort --path person_set/training/neg negative.bin
```
This should yield a `positive.bin` positive features file and a `negative.bin` negative features file in your current directory. The next step is to train the HOG model:
```
hog_trainer --pos positive.bin --neg negative.bin person_model.xml
```
`hog_trainer` should run for a while and then yield a `person_model.xml` trained SVM model file in your current directory, trained with the default HOG Trainer settings. The last step is to print statistics for the trained model and test its performance:
```
hog_run --pos person_set/testing/pos --neg person_set/testing/neg person_model.xml
```
`hog_run` actually runs HOG classification against the images in the positive and negative testing sets, and prints the percentage of images in each set that were classified correctly (recall/accuracy). If the output of `hog_run` is acceptable, then the job is done. Otherwise, `hog_trainer` can be re-run with different settings or with auto-training.

###`hog_snort`
This utility expects its images to all be the same size; it may skew images that deviate from the size it is given. `hog_snort` processes images serially and therefore takes up very little memory. It can be efficient to run `hog_snort` on a less powerful workstation (after image conversion and sorting) and then push the binary feature files to a more powerful computer that will do the training with `hog_trainer`.

###`hog_trainer`
An `--auto` argument is available to enable auto-training, which automatically selects the variables for the given kernel that give the best results. This is based on the CvSVM `train_auto` function. It is worth noting that the auto-training process takes a very long time, and may crash on extremely large (>14,000 examples) image sets.

###`hog_run`
This utility also expects its images to be the same size; it currently does not support automatic random sampling from negative test images, so those too must be the same size as the positive test images (which should in turn be the same size as the positive training set).

*Copyright (c) 2015 [University of Nevada, Las Vegas]*

[1]: http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
[University of Nevada, Las Vegas]: http://www.unlv.edu/
