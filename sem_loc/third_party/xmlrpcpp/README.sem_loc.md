# sem_loc XML-RPC retry fix

Source: https://github.com/ros/ros_comm/tree/1.17.4/utilities/xmlrpcpp
Pinned version: 1.17.4 (matches the installed ROS Noetic xmlrpcpp package).
XML-RPC license: LGPL-2.1, see COPYING; base64 license: libb64/LICENSE.

Only src/XmlRpcServer.cpp differs from upstream:

1. Clear _accept_error when re-arming the listener, so repeated work() calls
   cannot register duplicate poll sources while waiting for a new connection.
2. Treat nonblocking accept EAGAIN/EWOULDBLOCK and EINTR as retryable readiness
   events, leaving the listener monitored rather than reporting a hard error.

The shared library keeps the upstream SONAME libxmlrpcpp.so and is emitted under
build/sem_loc_ros_lib. Only sem_loc executables link to this directory via RPATH;
no file under /opt/ros is changed. Keep this directory with the executables.

Regression: test_xmlrpc_retry verifies nonblocking accept plus 300 work() cycles
across three simulated accept retries with a 64-FD soft limit. Unpatched code
reports EAGAIN and grows poll sources until poll returns EINVAL; patched code
keeps exactly one listener. The existing BEV integration suite exercises real
ROS master registration, subscriptions, publication and shutdown.
