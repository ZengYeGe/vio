#include "vio_app.hpp"

int PrintCommandUsage();

int main(int argc, char **argv) {

  Options option;
  // Required and optional arguments.
  for (int i = 0; i < argc; ++i) {
    if (!strcmp(argv[i], "--keyframe")) {
      option.use_keyframe = true;
      cout << "Using keyframe.\n";
    } else if (!strcmp(argv[i], "--config")) {
      option.config_filename = argv[++i];
    }
  }

  vio::VisualOdometryConfig vo_config;

  // Must have configuration for pipeline.
  if (option.config_filename.empty() ||
      !vo_config.SetUpFromFile(option.config_filename))
    return PrintCommandUsage();

  // Determine test type.
  if (argc < 2 || strcmp(argv[1], "--type"))
    return PrintCommandUsage();

  // ----------------------------Test images from dataset ---------------
  if (!strcmp(argv[2], "dataset")) {
    for (int i = 3; i < argc; ++i) {
      if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
        option.path = argv[++i];
      } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
        option.format = argv[++i];
      }
    }

    if (option.path.empty() || option.format.empty())
      return PrintCommandUsage();

    // Start to test
    TestFramesInFolder(option, vo_config);

  // ----------------------------Test video file ------------------------
  } else if (!strcmp(argv[2], "video")) {
    for (int i = 3; i < argc; ++i) {
      if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
        option.path = argv[++i];
      } else if (!strcmp(argv[i], "-c") ||
                 !strcmp(argv[i], "--calibration_file")) {
        option.calibration_filename = argv[++i];
      }
    }

    if (option.path.empty() || option.calibration_filename.empty())
      return PrintCommandUsage();

    if (!vo_config.SetUpCameraFromFile(option.calibration_filename))
      return -1;

    // Start to test
    TestVideo(option, vo_config);

  } else {
    return PrintCommandUsage();
  }

//  if (option.match_file_name.size())
//    return TestTwoFrameWithAccurateMatchFile(option);
  return 0;
}

int PrintCommandUsage() {
  cout << "Error. Unknown arguments.\n";
  cout << "Usage: ./vio_app_test --type [dataset] | [video] | [camera]\n";
  cout << "\n    For [dataset]:\n";
  cout << "            -p, --path : path to dataset \n";
  cout << "            -f, --format : image format of the dataset, e.g png, jpg\n";
  cout << "\n    For [video] :\n";
  cout << "            -p, --path : path to video file.\n";
  cout << "            -c, --calibration_file : path to camera config file.\n";
  cout << "\n    Other Required options:\n";
  cout << "            --config : path and name of configuration file.\n"; 
  cout << "\n    Other Optional options:\n";
  cout << "            --keyframe, select key frame\n";
  return -1;
}


