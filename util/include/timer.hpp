#include <sys/time.h>
#include <inttypes.h>

class Timer {
 public:
  Timer() {};

  void Start() { gettimeofday(&tv_start, NULL); }

  void Stop() { gettimeofday(&tv_end, NULL); }

  int64_t GetInMs() {
    const int64_t seconds = (tv_end.tv_sec - tv_start.tv_sec) * 1000;
    const int64_t milliseconds = (tv_end.tv_usec - tv_start.tv_usec) / 1000;
    return seconds + milliseconds;  
  }

 private:
  timeval tv_start;
  timeval tv_end;
};
