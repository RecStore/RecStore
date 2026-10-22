#include <pybind11/pybind11.h>

#include "recstore_kv_bindings.h"

// The engine registers two primitive gflags. Hidden no-op constructors keep
// the extension preloadable in processes that already embed another gflags.
extern "C" {
__attribute__((visibility("hidden"))) void
_ZN6google14FlagRegistererC1IiEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, int*, int*) {}
__attribute__((visibility("hidden"))) void
_ZN6google14FlagRegistererC1IdEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, double*, double*) {}
}

PYBIND11_MODULE(_recstore_kv, module) {
  module.attr("RECSTORE_VERSION")    = RECSTORE_VERSION;
  module.attr("BINDING_API_VERSION") = RECSTORE_KV_API_VERSION;
  module.attr("BINDING_ABI_VERSION") = RECSTORE_KV_ABI_VERSION;

  recstore::python::BindLocalKV(module);
  recstore::python::BindLocalIndexedKV(module);
}
