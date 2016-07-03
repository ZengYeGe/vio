alias clang-format='clang-format-3.6'

clang-format -style=file -i feature_tracker/include/*.hpp
clang-format -style=file -i feature_tracker/src/*.cpp

clang-format -style=file -i camera_model/include/*.hpp
clang-format -style=file -i camera_model/src/*.cpp

clang-format -style=file -i graph_optimizer/include/*.hpp
clang-format -style=file -i graph_optimizer/src/*.cpp

clang-format -style=file -i map_initializer/include/*.hpp
clang-format -style=file -i map_initializer/src/*.cpp

clang-format -style=file -i mapdata/include/*.hpp
clang-format -style=file -i mapdata/src/*.cpp

clang-format -style=file -i multiview_helper/include/*.hpp
clang-format -style=file -i multiview_helper/src/*.cpp

clang-format -style=file -i pnp_estimator/include/*.hpp
clang-format -style=file -i pnp_estimator/src/*.cpp

clang-format -style=file -i util/include/*.hpp
clang-format -style=file -i util/src/*.cpp

clang-format -style=file -i visual_odometry/include/*.hpp
clang-format -style=file -i visual_odometry/src/*.cpp

clang-format -style=file -i vio_app/include/*.hpp
clang-format -style=file -i vio_app/src/*.cpp


