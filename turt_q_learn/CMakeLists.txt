cmake_minimum_required(VERSION 2.8.3)
project(turt_q_learn)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
)


catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
