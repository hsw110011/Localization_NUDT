# CMake generated Testfile for 
# Source directory: /home/hsw/catkin_ws/program/sem_loc
# Build directory: /home/hsw/catkin_ws/program/sem_loc/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(xmlrpc_retry "/home/hsw/catkin_ws/program/sem_loc/build/devel/lib/loc_bev/test_xmlrpc_retry")
set_tests_properties(xmlrpc_retry PROPERTIES  _BACKTRACE_TRIPLES "/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;173;add_test;/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;0;")
add_test(motion2d "/home/hsw/catkin_ws/program/sem_loc/build/devel/lib/loc_bev/test_motion2d")
set_tests_properties(motion2d PROPERTIES  _BACKTRACE_TRIPLES "/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;176;add_test;/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;0;")
add_test(particle_motion "/home/hsw/catkin_ws/program/sem_loc/build/devel/lib/loc_bev/test_particle_motion")
set_tests_properties(particle_motion PROPERTIES  _BACKTRACE_TRIPLES "/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;180;add_test;/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;0;")
add_test(scoring_routes "/home/hsw/catkin_ws/program/sem_loc/build/devel/lib/loc_bev/test_scoring_routes")
set_tests_properties(scoring_routes PROPERTIES  _BACKTRACE_TRIPLES "/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;185;add_test;/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;0;")
add_test(bev_config "/usr/bin/python3" "/home/hsw/catkin_ws/program/sem_loc/tests/bev_config_test.py" "/home/hsw/catkin_ws/program/sem_loc/build/bev")
set_tests_properties(bev_config PROPERTIES  _BACKTRACE_TRIPLES "/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;186;add_test;/home/hsw/catkin_ws/program/sem_loc/CMakeLists.txt;0;")
subdirs("gtest")
