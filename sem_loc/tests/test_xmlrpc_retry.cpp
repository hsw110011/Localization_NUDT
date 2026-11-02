#include <xmlrpcpp/XmlRpcServer.h>
#include <xmlrpcpp/XmlRpcUtil.h>
#include <sys/resource.h>
#include <iostream>
#include <stdexcept>

class RetryServer : public XmlRpc::XmlRpcServer {
public:
    // Mirror dispatcher state after a real accept failure removes the listener.
    void simulateAcceptFailure() {
        _disp.removeSource(this);
        _accept_error = true;
        _accept_retry_time_sec = 0;
    }
    unsigned acceptWithoutClient() { return acceptConnection(); }
    size_t sourceCount() const { return _disp._sources.size(); }
};
class ErrorCounter : public XmlRpc::XmlRpcErrorHandler {
public:
    int count = 0;
    void error(const char* message) override { ++count; std::cerr << message << '\n'; }
};
int main() {
    rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) throw std::runtime_error("getrlimit failed");
    limit.rlim_cur = 64;
    if (setrlimit(RLIMIT_NOFILE, &limit) != 0) throw std::runtime_error("setrlimit failed");
    ErrorCounter errors;
    auto* original = XmlRpc::XmlRpcErrorHandler::getErrorHandler();
    XmlRpc::XmlRpcErrorHandler::setErrorHandler(&errors);
    {
        RetryServer server;
        if (!server.bindAndListen(0)) throw std::runtime_error("bind failed");
        if (server.acceptWithoutClient() != XmlRpc::XmlRpcDispatch::ReadableEvent || errors.count)
            throw std::runtime_error("EAGAIN must leave the listener monitored without an error");
        for (int retry = 0; retry < 3; ++retry) {
            server.simulateAcceptFailure();
            for (int i = 0; i < 100; ++i) server.work(.001);
            if (server.sourceCount() != 1 || errors.count)
                throw std::runtime_error("accept retry duplicated listener / caused poll failure");
        }
    }
    XmlRpc::XmlRpcErrorHandler::setErrorHandler(original);
    std::cout << "Nonblocking accept and 300 retry iterations under 64-FD limit: OK\n";
}
