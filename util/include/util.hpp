#ifdef __linux__
#include <dirent.h>
#endif

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

bool GetImageNamesInFolder(const std::string &path, const std::string &format,
                           std::vector<std::string> &images);

