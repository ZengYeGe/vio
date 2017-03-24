** Build
mkdir build
cd build
cmake ..
make

** Command

./vio_app/vio_app_test --type video -p ../data/record_fast.avi -c ../data/recon2v_checkerboards.txt --config ../data/config.yaml

./vio_app/vio_app_test --type dataset -p ../data/temple/ -f png --config ../data/config.yaml

** Before push
*** Use clang format
*** Check if needed to update the system flow graph


** TODO
1) CI integration, e.g. Travis
2) ROS integration
3) Android integration, currently vio_ros works with Android ros_package



