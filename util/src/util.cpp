#include "util.hpp"

bool GetImageNamesInFolder(const std::string &path, const std::string &format,
                           std::vector<std::string> &images) {
  struct dirent **file_list;
  int n = scandir(path.c_str(), &file_list, 0, alphasort);
  if (n < 0) {
    std::cerr << "Error: Unable to find directory " << path << std::endl;
    return false;
  } else {
    for (int i = 0; i < n; ++i) {
      std::string file_name(file_list[i]->d_name);
      if (file_name.size() > format.size() &&
          !file_name.compare(file_name.size() - format.size(),
                             format.size(), format)) {
        images.push_back(path + '/' + file_name);
      }
    }
  }

  free(file_list);
  return true;
}

