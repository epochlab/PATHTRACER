// Strict ISO C11: proves pathtracer_c.h compiles for a C caller and reports the PT_ABI_VERSION that caller's header carries.

#include "pathtracer/api/pathtracer_c.h"

int ptHeaderAbiVersion(void) {
    return PT_ABI_VERSION;
}
