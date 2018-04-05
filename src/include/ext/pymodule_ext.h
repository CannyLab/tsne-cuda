
#ifndef PYMODULE_EXT_H
#define PYMODULE_EXT_H

    #include <sys/types.h>
    #include "common.h"
    #include "util/distance_utils.h"

    extern "C" {
        void pymodule_e_dist(float* points, float* dist, ssize_t *dims);
    }

#endif