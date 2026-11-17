#include "storage/io_backend/force_link.h"
#include "storage/kv_engine/engine_composite.h"
#include "storage/kv_engine/engine_petkv.h"

#include "gflags/gflags.h"

extern "C" void RecStoreForceLinkFasterKVEngine();
extern "C" void RecStoreForceLinkDETableEngine();
extern "C" void RecStoreForceLinkHPSEngine();

namespace {
struct IOBackendLinkGuard {
  IOBackendLinkGuard() {
    ForceLinkIOBackends();
#ifdef RECSTORE_ENABLE_DETABLE_ENGINE
    RecStoreForceLinkDETableEngine();
#endif
#ifdef RECSTORE_ENABLE_FASTERKV_ENGINE
    RecStoreForceLinkFasterKVEngine();
#endif
#ifdef RECSTORE_ENABLE_HPS_ENGINE
    RecStoreForceLinkHPSEngine();
#endif
  }
};
const IOBackendLinkGuard kIoBackendLinkGuard;
} // namespace

// Explicit link entry point for static builds.
//
// This translation unit only holds factory registrations and one static
// initializer, so a static library drops it unless something references a
// symbol defined here -- and then every BaseKV / index / value-store /
// allocator lookup fails at runtime with "no Factory instance". Shared
// builds do not need it (`--no-as-needed` keeps the library loaded), so this
// has to be called explicitly by executables that link `kv_engine` statically.
extern "C" void RecStoreForceLinkKVEngineRegistrations() {}

DEFINE_int32(prefetch_method,
             0,
             "PetKV BatchGet prefetch method: 0=single Get loop, 1=BatchGet");
