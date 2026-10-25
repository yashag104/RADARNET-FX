#ifndef RN_INFER_H
#define RN_INFER_H
#include <stdint.h>
#include "rn_model.h"

/* in_iq: channel-major INT8 input, [0..127] = I, [128..255] = Q.
 * Writes the INT32 logits and returns the predicted class (first maximum). */
int rn_infer(const int8_t in_iq[256], int32_t logits[RN_NUM_CLASSES]);

#endif
