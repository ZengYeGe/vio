#include "vio_app.hpp"

int TestTwoFrameWithAccurateMatchFile(Options options) {
  cv::Mat_<double> x1, x2;
  int npts;
  ifstream myfile(options.match_file_name.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << options.match_file_name << endl;
    exit(0);
  }
  vector<vector<cv::Vec2d> > feature_vectors(2);

  string line;

  // Read number of points
  getline(myfile, line);
  npts = (int)atof(line.c_str());

  feature_vectors[0].resize(npts);
  feature_vectors[1].resize(npts);

  x1 = Mat_<double>(2, npts);
  x2 = Mat_<double>(2, npts);

  // Read the point coordinates
  for (int i = 0; i < npts; ++i) {
    getline(myfile, line);
    stringstream s(line);
    string cord;

    s >> cord;
    x1(0, i) = atof(cord.c_str());
    s >> cord;
    x1(1, i) = atof(cord.c_str());

    s >> cord;
    x2(0, i) = atof(cord.c_str());
    s >> cord;
    x2(1, i) = atof(cord.c_str());

    feature_vectors[0][i][0] = x1(0, i);
    feature_vectors[0][i][1] = x1(1, i);
    feature_vectors[1][i][0] = x2(0, i);
    feature_vectors[1][i][1] = x2(1, i);

  }
  myfile.close();

  RunInitializer(feature_vectors);

  return 0;
}

