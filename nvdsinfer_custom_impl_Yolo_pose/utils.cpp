#include "utils.h"

float
clamp(const float val, const float minVal, const float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}
