#include "vio_app.hpp"

int main(int argc, char **argv) {
  Options option;
  for (int i = 0; i < argc; ++i) {
    if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
      option.path = argv[++i];
    } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
      option.format = argv[++i];
    } else if (!strcmp(argv[i], "--keyframe")) {
      option.use_keyframe = true;
      cout << "Using keyframe.\n";
    } else if (!strcmp(argv[i], "--accuratetwo")) {
      option.match_file_name = argv[++i];
    }
  }

  if (option.match_file_name.size())
    return TestTwoFrameWithAccurateMatchFile(option);

  if (option.format.size() && option.path.size())
    return TestFramesInFolder(option);

  cout << "Error. Unknown arguments.\n";
  cout << "Usage: \n";
  cout << "       test\n";
  cout << "            -p, --path full_path \n";
  cout << "            -f, --format image format, e.g png, jpg\n";
  cout << "            --keyframe, select key frame\n";
  // TODO: Add test two frames with good baseline and bad baseline.
  cout << "            --accuratetwo  path_to_match_file, test using accurate "
          "matches.\n";

  return -1;
}

