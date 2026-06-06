execute_process(COMMAND "/home/hsw/catkin_ws/program/dem_loc/build/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/hsw/catkin_ws/program/dem_loc/build/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
